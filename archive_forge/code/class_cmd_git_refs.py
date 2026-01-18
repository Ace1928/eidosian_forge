import breezy.bzr  # noqa: F401
from breezy import controldir
from ..commands import Command, display_command
from ..option import Option, RegistryOption
class cmd_git_refs(Command):
    """Output all of the virtual refs for a repository.

    """
    hidden = True
    takes_args = ['location?']

    @display_command
    def run(self, location='.'):
        from ..controldir import ControlDir
        from .object_store import get_object_store
        from .refs import get_refs_container
        controldir, _ = ControlDir.open_containing(location)
        repo = controldir.find_repository()
        object_store = get_object_store(repo)
        with object_store.lock_read():
            refs = get_refs_container(controldir, object_store)
            for k, v in sorted(refs.as_dict().items()):
                self.outf.write('%s -> %s\n' % (k.decode('utf-8'), v.decode('utf-8')))