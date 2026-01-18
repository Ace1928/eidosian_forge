from .. import transport
class GitHubDirectory:

    def look_up(self, name, url, purpose=None):
        """See DirectoryService.look_up"""
        return 'git+ssh://git@github.com/' + name