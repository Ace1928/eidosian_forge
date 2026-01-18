import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from ... import bedding
from ... import branch as _mod_branch
from ... import controldir, errors, hooks, urlutils
from ... import version_string as breezy_version
from ...config import AuthenticationConfig, GlobalStack
from ...errors import (InvalidHttpResponse, PermissionDenied,
from ...forge import (Forge, ForgeLoginRequired, MergeProposal,
from ...git.urls import git_url_to_bzr_url
from ...i18n import gettext
from ...trace import note
from ...transport import get_transport
from ...transport.http import default_user_agent
class GitHub(Forge):
    name = 'github'
    supports_merge_proposal_labels = True
    supports_merge_proposal_commit_message = False
    supports_merge_proposal_title = True
    supports_allow_collaboration = True
    merge_proposal_description_format = 'markdown'

    def __repr__(self):
        return 'GitHub()'

    def _graphql_request(self, body, **kwargs):
        headers = {}
        if self._token:
            headers['Authorization'] = 'token %s' % self._token
        url = urlutils.join(self.transport.base, 'graphql')
        response = self.transport.request('POST', url, headers=headers, body=json.dumps({'query': body, 'variables': kwargs}).encode('utf-8'))
        if response.status != 200:
            raise UnexpectedHttpStatus(url, response.status, headers=response.getheaders())
        data = json.loads(response.text)
        if data.get('errors'):
            raise GraphqlErrors(data.get('errors'))
        return data['data']

    def _api_request(self, method, path, body=None):
        headers = {'Content-Type': 'application/json', 'Accept': 'application/vnd.github.v3+json'}
        if self._token:
            headers['Authorization'] = 'token %s' % self._token
        try:
            response = self.transport.request(method, urlutils.join(self.transport.base, path), headers=headers, body=body, retries=3)
        except UnexpectedHttpStatus as e:
            if e.code == 401:
                raise GitHubLoginRequired(self.base_url)
            else:
                raise
        if response.status == 401:
            raise GitHubLoginRequired(self.base_url)
        return response

    def _get_repo(self, owner, repo):
        path = 'repos/{}/{}'.format(owner, repo)
        response = self._api_request('GET', path)
        if response.status == 404:
            raise NoSuchProject(path)
        if response.status == 200:
            return json.loads(response.text)
        raise UnexpectedHttpStatus(path, response.status, headers=response.getheaders())

    def _get_repo_pulls(self, path, head=None, state=None):
        path = path + '?'
        params = {}
        if head is not None:
            params['head'] = head
        if state is not None:
            params['state'] = state
        path += ';'.join(['{}={}'.format(k, urlutils.quote(v)) for k, v in params.items()])
        response = self._api_request('GET', path)
        if response.status == 404:
            raise NoSuchProject(path)
        if response.status == 200:
            return json.loads(response.text)
        raise UnexpectedHttpStatus(path, response.status, headers=response.getheaders())

    def _create_pull(self, path, title, head, base, body=None, labels=None, assignee=None, draft=False, maintainer_can_modify=False):
        data = {'title': title, 'head': head, 'base': base, 'draft': draft, 'maintainer_can_modify': maintainer_can_modify}
        if labels is not None:
            data['labels'] = labels
        if assignee is not None:
            data['assignee'] = assignee
        if body:
            data['body'] = body
        response = self._api_request('POST', path, body=json.dumps(data).encode('utf-8'))
        if response.status == 403:
            raise PermissionDenied(path, response.text)
        if response.status == 422:
            raise ValidationFailed(json.loads(response.text))
        if response.status != 201:
            raise UnexpectedHttpStatus(path, response.status, headers=response.getheaders())
        return json.loads(response.text)

    def _get_user_by_email(self, email):
        path = 'search/users?q=%s+in:email' % email
        response = self._api_request('GET', path)
        if response.status != 200:
            raise UnexpectedHttpStatus(path, response.status, headers=response.getheaders())
        ret = json.loads(response.text)
        if ret['total_count'] == 0:
            raise KeyError('no user with email %s' % email)
        elif ret['total_count'] > 1:
            raise ValueError('more than one result for email %s' % email)
        return ret['items'][0]

    def _get_user(self, username=None):
        if username:
            path = 'users/%s' % username
        else:
            path = 'user'
        response = self._api_request('GET', path)
        if response.status != 200:
            raise UnexpectedHttpStatus(path, response.status, headers=response.getheaders())
        return json.loads(response.text)

    def _get_organization(self, name):
        path = 'orgs/%s' % name
        response = self._api_request('GET', path)
        if response.status != 200:
            raise UnexpectedHttpStatus(path, response.status, headers=response.getheaders())
        return json.loads(response.text)

    def _list_paged(self, path, parameters=None, per_page=None):
        if parameters is None:
            parameters = {}
        else:
            parameters = dict(parameters.items())
        if per_page:
            parameters['per_page'] = str(per_page)
        page = 1
        while path:
            parameters['page'] = str(page)
            response = self._api_request('GET', path + '?' + ';'.join(['{}={}'.format(k, urlutils.quote(v)) for k, v in parameters.items()]))
            if response.status != 200:
                raise UnexpectedHttpStatus(path, response.status, headers=response.getheaders())
            data = json.loads(response.text)
            if not data:
                break
            yield data
            page += 1

    def _search_issues(self, query):
        path = 'search/issues'
        for page in self._list_paged(path, {'q': query}, per_page=DEFAULT_PER_PAGE):
            if not page['items']:
                break
            yield from page['items']

    def _create_fork(self, path, owner=None):
        if owner and owner != self.current_user['login']:
            path += '?organization=%s' % owner
        response = self._api_request('POST', path)
        if response.status != 202:
            raise UnexpectedHttpStatus(path, response.status, headers=response.getheaders())
        return json.loads(response.text)

    @property
    def base_url(self):
        return WEB_GITHUB_URL

    def __init__(self, transport):
        self._token = retrieve_github_token()
        if self._token is None:
            note("Accessing GitHub anonymously. To log in, run 'brz gh-login'.")
        self.transport = transport
        self._current_user = None

    @property
    def current_user(self):
        if self._current_user is None:
            self._current_user = self._get_user()
        return self._current_user

    def publish_derived(self, local_branch, base_branch, name, project=None, owner=None, revision_id=None, overwrite=False, allow_lossy=True, tag_selector=None):
        if tag_selector is None:
            tag_selector = lambda t: False
        base_owner, base_project, base_branch_name = parse_github_branch_url(base_branch)
        base_repo = self._get_repo(base_owner, base_project)
        if owner is None:
            owner = self.current_user['login']
        if project is None:
            project = base_repo['name']
        try:
            remote_repo = self._get_repo(owner, project)
        except NoSuchProject:
            base_repo = self._get_repo(base_owner, base_project)
            remote_repo = self._create_fork(base_repo['forks_url'], owner)
            note(gettext('Forking new repository %s from %s') % (remote_repo['html_url'], base_repo['html_url']))
        else:
            note(gettext('Reusing existing repository %s') % remote_repo['html_url'])
        remote_dir = controldir.ControlDir.open(git_url_to_bzr_url(remote_repo['ssh_url']))
        try:
            push_result = remote_dir.push_branch(local_branch, revision_id=revision_id, overwrite=overwrite, name=name, tag_selector=tag_selector)
        except errors.NoRoundtrippingSupport:
            if not allow_lossy:
                raise
            push_result = remote_dir.push_branch(local_branch, revision_id=revision_id, overwrite=overwrite, name=name, lossy=True, tag_selector=tag_selector)
        return (push_result.target_branch, github_url_to_bzr_url(remote_repo['clone_url'], name))

    def get_push_url(self, branch):
        owner, project, branch_name = parse_github_branch_url(branch)
        repo = self._get_repo(owner, project)
        return github_url_to_bzr_url(repo['ssh_url'], branch_name)

    def get_web_url(self, branch):
        owner, project, branch_name = parse_github_branch_url(branch)
        repo = self._get_repo(owner, project)
        if branch_name:
            return repo['html_url'] + '/tree/' + branch_name
        else:
            return repo['html_url']

    def get_derived_branch(self, base_branch, name, project=None, owner=None, preferred_schemes=None):
        base_owner, base_project, base_branch_name = parse_github_branch_url(base_branch)
        base_repo = self._get_repo(base_owner, base_project)
        if owner is None:
            owner = self.current_user['login']
        if project is None:
            project = base_repo['name']
        try:
            remote_repo = self._get_repo(owner, project)
        except NoSuchProject:
            raise errors.NotBranchError('{}/{}/{}'.format(WEB_GITHUB_URL, owner, project))
        if preferred_schemes is None:
            preferred_schemes = DEFAULT_PREFERRED_SCHEMES
        for scheme in preferred_schemes:
            if scheme in SCHEME_FIELD_MAP:
                github_url = remote_repo[SCHEME_FIELD_MAP[scheme]]
                break
        else:
            raise AssertionError
        full_url = github_url_to_bzr_url(github_url, name)
        return _mod_branch.Branch.open(full_url)

    def get_proposer(self, source_branch, target_branch):
        return GitHubMergeProposalBuilder(self, source_branch, target_branch)

    def iter_proposals(self, source_branch, target_branch, status='open'):
        source_owner, source_repo_name, source_branch_name = parse_github_branch_url(source_branch)
        target_owner, target_repo_name, target_branch_name = parse_github_branch_url(target_branch)
        target_repo = self._get_repo(target_owner, target_repo_name)
        state = {'open': 'open', 'merged': 'closed', 'closed': 'closed', 'all': 'all'}
        pulls = self._get_repo_pulls(strip_optional(target_repo['pulls_url']), head=target_branch_name, state=state[status])
        for pull in pulls:
            if status == 'closed' and pull['merged'] or (status == 'merged' and (not pull['merged'])):
                continue
            if pull['head']['ref'] != source_branch_name:
                continue
            if pull['head']['repo'] is None:
                continue
            if pull['head']['repo']['owner']['login'] != source_owner or pull['head']['repo']['name'] != source_repo_name:
                continue
            yield GitHubMergeProposal(self, pull)

    def hosts(self, branch):
        try:
            parse_github_branch_url(branch)
        except NotGitHubUrl:
            return False
        else:
            return True

    @classmethod
    def probe_from_hostname(cls, hostname, possible_transports=None):
        if hostname == GITHUB_HOST:
            transport = get_transport(API_GITHUB_URL, possible_transports=possible_transports)
            return cls(transport)
        raise UnsupportedForge(hostname)

    @classmethod
    def probe_from_url(cls, url, possible_transports=None):
        try:
            parse_github_url(url)
        except NotGitHubUrl:
            raise UnsupportedForge(url)
        transport = get_transport(API_GITHUB_URL, possible_transports=possible_transports)
        return cls(transport)

    @classmethod
    def iter_instances(cls):
        yield cls(get_transport(API_GITHUB_URL))

    def iter_my_proposals(self, status='open', author=None):
        query = ['is:pr']
        if status == 'open':
            query.append('is:open')
        elif status == 'closed':
            query.append('is:unmerged')
            query.append('is:closed')
        elif status == 'merged':
            query.append('is:merged')
        if author is None:
            author = self.current_user['login']
        query.append('author:%s' % author)
        for issue in self._search_issues(query=' '.join(query)):

            def retrieve_full():
                response = self._api_request('GET', issue['pull_request']['url'])
                if response.status != 200:
                    raise UnexpectedHttpStatus(issue['pull_request']['url'], response.status, headers=response.getheaders())
                return json.loads(response.text)
            yield GitHubMergeProposal(self, _LazyDict(issue['pull_request'], retrieve_full))

    def get_proposal_by_url(self, url):
        try:
            owner, repo, pr_id = parse_github_pr_url(url)
        except NotGitHubUrl as e:
            raise UnsupportedForge(url) from e
        api_url = 'https://api.github.com/repos/{}/{}/pulls/{}'.format(owner, repo, pr_id)
        response = self._api_request('GET', api_url)
        if response.status != 200:
            raise UnexpectedHttpStatus(api_url, response.status, headers=response.getheaders())
        data = json.loads(response.text)
        return GitHubMergeProposal(self, data)

    def iter_my_forks(self, owner=None):
        if owner:
            path = '/users/%s/repos' % owner
        else:
            path = '/user/repos'
        for page in self._list_paged(path, per_page=DEFAULT_PER_PAGE):
            for project in page:
                if not project['fork']:
                    continue
                yield project['full_name']

    def delete_project(self, path):
        path = 'repos/' + path
        response = self._api_request('DELETE', path)
        if response.status == 404:
            raise NoSuchProject(path)
        if response.status == 204:
            return
        if response.status == 200:
            return json.loads(response.text)
        raise UnexpectedHttpStatus(path, response.status, headers=response.getheaders())

    def create_project(self, path, *, homepage=None, private=False, has_issues=True, has_projects=False, has_wiki=False, summary=None):
        owner, name = path.split('/')
        path = 'repos'
        data = {'name': 'name', 'description': summary, 'homepage': homepage, 'private': private, 'has_issues': has_issues, 'has_projects': has_projects, 'has_wiki': has_wiki}
        response = self._api_request('POST', path, body=json.dumps(data).encode('utf-8'))
        if response.status != 201:
            return json.loads(response.text)
        raise UnexpectedHttpStatus(path, response.status, headers=response.getheaders())

    def get_current_user(self):
        if self._token is not None:
            return self.current_user['login']
        return None

    def get_user_url(self, username):
        return urlutils.join(self.base_url, username)