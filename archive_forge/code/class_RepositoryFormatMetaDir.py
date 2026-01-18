import itertools
from .. import errors, lockable_files, lockdir
from .. import revision as _mod_revision
from .. import transport as _mod_transport
from ..repository import Repository, RepositoryFormat, format_registry
from . import bzrdir
class RepositoryFormatMetaDir(bzrdir.BzrFormat, RepositoryFormat):
    """Common base class for the new repositories using the metadir layout."""
    rich_root_data = False
    supports_tree_reference = False
    supports_external_lookups = False
    supports_leaving_lock = True
    supports_nesting_repositories = True

    @property
    def _matchingcontroldir(self):
        matching = bzrdir.BzrDirMetaFormat1()
        matching.repository_format = self
        return matching

    def __init__(self):
        RepositoryFormat.__init__(self)
        bzrdir.BzrFormat.__init__(self)

    def _create_control_files(self, a_bzrdir):
        """Create the required files and the initial control_files object."""
        repository_transport = a_bzrdir.get_repository_transport(self)
        control_files = lockable_files.LockableFiles(repository_transport, 'lock', lockdir.LockDir)
        control_files.create_lock()
        return control_files

    def _upload_blank_content(self, a_bzrdir, dirs, files, utf8_files, shared):
        """Upload the initial blank content."""
        control_files = self._create_control_files(a_bzrdir)
        control_files.lock_write()
        transport = control_files._transport
        if shared is True:
            utf8_files += [('shared-storage', b'')]
        try:
            for dir in dirs:
                transport.mkdir(dir, mode=a_bzrdir._get_dir_mode())
            for filename, content_stream in files:
                transport.put_file(filename, content_stream, mode=a_bzrdir._get_file_mode())
            for filename, content_bytes in utf8_files:
                transport.put_bytes_non_atomic(filename, content_bytes, mode=a_bzrdir._get_file_mode())
        finally:
            control_files.unlock()

    @classmethod
    def find_format(klass, a_bzrdir):
        """Return the format for the repository object in a_bzrdir.

        This is used by brz native formats that have a "format" file in
        the repository.  Other methods may be used by different types of
        control directory.
        """
        try:
            transport = a_bzrdir.get_repository_transport(None)
            format_string = transport.get_bytes('format')
        except _mod_transport.NoSuchFile:
            raise errors.NoRepositoryPresent(a_bzrdir)
        return klass._find_format(format_registry, 'repository', format_string)

    def check_support_status(self, allow_unsupported, recommend_upgrade=True, basedir=None):
        RepositoryFormat.check_support_status(self, allow_unsupported=allow_unsupported, recommend_upgrade=recommend_upgrade, basedir=basedir)
        bzrdir.BzrFormat.check_support_status(self, allow_unsupported=allow_unsupported, recommend_upgrade=recommend_upgrade, basedir=basedir)