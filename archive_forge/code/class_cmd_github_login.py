from ... import errors
from ...commands import Command
class cmd_github_login(Command):
    __doc__ = 'Log into GitHub.\n\n    When communicating with GitHub, some commands need to authenticate to\n    GitHub.\n    '
    takes_args = ['username?']

    def run(self, username=None):
        from github import Github, GithubException
        from breezy.config import AuthenticationConfig
        authconfig = AuthenticationConfig()
        if username is None:
            username = authconfig.get_user('https', 'github.com', prompt='GitHub username', ask=True)
        password = authconfig.get_password('https', 'github.com', username)
        client = Github(username, password)
        user = client.get_user()
        try:
            authorization = user.create_authorization(scopes=['user', 'repo', 'delete_repo'], note='Breezy', note_url='https://github.com/breezy-team/breezy')
        except GithubException as e:
            errs = e.data.get('errors', [])
            if errs:
                err_code = errs[0].get('code')
                if err_code == 'already_exists':
                    raise errors.CommandError('token already exists')
            raise errors.CommandError(e.data['message'])
        from .forge import store_github_token
        store_github_token(token=authorization.token)