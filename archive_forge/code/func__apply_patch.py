import breezy.bzr  # noqa: F401
from breezy import controldir
from ..commands import Command, display_command
from ..option import Option, RegistryOption
def _apply_patch(self, wt, f, signoff):
    """Apply a patch.

        :param wt: A Bazaar working tree object.
        :param f: Patch file to read.
        :param signoff: Add Signed-Off-By flag.
        """
    from dulwich.patch import git_am_patch_split
    from breezy.patch import patch_tree
    c, diff, version = git_am_patch_split(f)
    patch_tree(wt, [diff], strip=1, out=self.outf)
    message = c.message.decode('utf-8')
    if signoff:
        signed_off_by = wt.branch.get_config().username()
        message += 'Signed-off-by: {}\n'.format(signed_off_by)
    wt.commit(authors=[c.author.decode('utf-8')], message=message)