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
class GitHubMergeProposal(MergeProposal):
    supports_auto_merge = True

    def __init__(self, gh, pr):
        self._gh = gh
        self._pr = pr

    def __repr__(self):
        return '<{} at {!r}>'.format(type(self).__name__, self.url)
    name = 'GitHub'

    def get_web_url(self):
        return self._pr['html_url']

    @property
    def url(self):
        return self._pr['html_url']

    def _branch_from_part(self, part, preferred_schemes=None):
        if part['repo'] is None:
            return None
        if preferred_schemes is None:
            preferred_schemes = DEFAULT_PREFERRED_SCHEMES
        for scheme in preferred_schemes:
            if scheme in SCHEME_FIELD_MAP:
                return github_url_to_bzr_url(part['repo'][SCHEME_FIELD_MAP[scheme]], part['ref'])
        raise AssertionError

    def get_source_branch_url(self, *, preferred_schemes=None):
        return self._branch_from_part(self._pr['head'], preferred_schemes=preferred_schemes)

    def get_source_revision(self):
        """Return the latest revision for the source branch."""
        from breezy.git.mapping import default_mapping
        return default_mapping.revision_id_foreign_to_bzr(self._pr['head']['sha'].encode('ascii'))

    def get_target_branch_url(self, *, preferred_schemes=None):
        return self._branch_from_part(self._pr['base'], preferred_schemes=preferred_schemes)

    def set_target_branch_name(self, name):
        self._patch(base=name)

    def get_source_project(self):
        if self._pr['head']['repo'] is None:
            return None
        return self._pr['head']['repo']['full_name']

    def get_target_project(self):
        if self._pr['base']['repo'] is None:
            return None
        return self._pr['base']['repo']['full_name']

    def get_description(self):
        return self._pr['body']

    def get_commit_message(self):
        return None

    def get_title(self):
        return self._pr.get('title')

    def set_title(self, title):
        self._patch(title=title)

    def set_commit_message(self, message):
        raise errors.UnsupportedOperation(self.set_commit_message, self)

    def _patch(self, **data):
        response = self._gh._api_request('PATCH', self._pr['url'], body=json.dumps(data).encode('utf-8'))
        if response.status == 422:
            raise ValidationFailed(json.loads(response.text))
        if response.status != 200:
            raise UnexpectedHttpStatus(self._pr['url'], response.status, headers=response.getheaders())
        self._pr = json.loads(response.text)

    def set_description(self, description):
        self._patch(body=description, title=determine_title(description))

    def is_merged(self):
        return bool(self._pr.get('merged_at'))

    def is_closed(self):
        return self._pr['state'] == 'closed' and (not bool(self._pr.get('merged_at')))

    def reopen(self):
        try:
            self._patch(state='open')
        except ValidationFailed as e:
            raise ReopenFailed(e.error['errors'][0]['message'])

    def close(self):
        self._patch(state='closed')

    def can_be_merged(self):
        return self._pr['mergeable']

    def merge(self, commit_message=None, auto=False):
        if auto:
            graphql_query = '\nmutation ($pullRequestId: ID!) {\n  enablePullRequestAutoMerge(input: {\n    pullRequestId: $pullRequestId,\n    mergeMethod: MERGE\n  }) {\n    pullRequest {\n      autoMergeRequest {\n        enabledAt\n        enabledBy {\n          login\n        }\n      }\n    }\n  }\n}\n'
            try:
                self._gh._graphql_request(graphql_query, pullRequestId=self._pr['node_id'])
            except GraphqlErrors as e:
                mutter('graphql errors: %r', e.errors)
                first_error = e.errors[0]
                if first_error['type'] == 'UNPROCESSABLE' and first_error['path'] == 'enablePullRequestAutoMerge':
                    raise Exception(first_error['message'])
                raise Exception(first_error['message'])
        else:
            data = {}
            if commit_message:
                data['commit_message'] = commit_message
            response = self._gh._api_request('PUT', self._pr['url'] + '/merge', body=json.dumps(data).encode('utf-8'))
            if response.status == 422:
                raise ValidationFailed(json.loads(response.text))
            if response.status != 200:
                raise UnexpectedHttpStatus(self._pr['url'], response.status, headers=response.getheaders())

    def get_merged_by(self):
        merged_by = self._pr.get('merged_by')
        if merged_by is None:
            return None
        return merged_by['login']

    def get_merged_at(self):
        merged_at = self._pr.get('merged_at')
        if merged_at is None:
            return None
        return parse_timestring(merged_at)

    def post_comment(self, body):
        data = {'body': body}
        response = self._gh._api_request('POST', self._pr['comments_url'], body=json.dumps(data).encode('utf-8'))
        if response.status == 422:
            raise ValidationFailed(json.loads(response.text))
        if response.status != 201:
            raise UnexpectedHttpStatus(self._pr['comments_url'], response.status, headers=response.getheaders())
        json.loads(response.text)