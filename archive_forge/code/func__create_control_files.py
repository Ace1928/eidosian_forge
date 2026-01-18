import itertools
from .. import errors, lockable_files, lockdir
from .. import revision as _mod_revision
from .. import transport as _mod_transport
from ..repository import Repository, RepositoryFormat, format_registry
from . import bzrdir
def _create_control_files(self, a_bzrdir):
    """Create the required files and the initial control_files object."""
    repository_transport = a_bzrdir.get_repository_transport(self)
    control_files = lockable_files.LockableFiles(repository_transport, 'lock', lockdir.LockDir)
    control_files.create_lock()
    return control_files