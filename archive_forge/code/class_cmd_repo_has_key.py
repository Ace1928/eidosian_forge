from ...commands import Command
from ...controldir import ControlDir
class cmd_repo_has_key(Command):
    """Does a repo have a key?

    e.g.::

      bzr repo-has-key texts FILE-ID REVISION-ID
      bzr repo-has-key revisions REVISION-ID

    It either prints "True" or "False", and terminates with exit code 0 or 1
    respectively.
    """
    hidden = True
    takes_args = ['repo', 'key_parts*']

    def run(self, repo, key_parts_list=None):
        vf_name, key = (key_parts_list[0], key_parts_list[1:])
        bd = ControlDir.open(repo)
        repo = bd.open_repository()
        with repo.lock_read():
            vf = getattr(repo, vf_name)
            key = tuple(key)
            if key in vf.get_parent_map([key]):
                self.outf.write('True\n')
                return 0
            else:
                self.outf.write('False\n')
                return 1