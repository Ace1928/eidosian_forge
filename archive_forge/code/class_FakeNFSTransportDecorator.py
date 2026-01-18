from stat import S_ISDIR
from .. import errors
from .. import transport as _mod_transport
from .. import urlutils
from . import decorator
class FakeNFSTransportDecorator(decorator.TransportDecorator):
    """A transport that behaves like NFS, for testing"""

    @classmethod
    def _get_url_prefix(self):
        """FakeNFS transports are identified by 'fakenfs+'"""
        return 'fakenfs+'

    def rename(self, rel_from, rel_to):
        """See Transport.rename().

        This variation on rename converts DirectoryNotEmpty and FileExists
        errors into ResourceBusy if the target is a directory.
        """
        try:
            self._decorated.rename(rel_from, rel_to)
        except (errors.DirectoryNotEmpty, _mod_transport.FileExists) as e:
            stat = self._decorated.stat(rel_to)
            if S_ISDIR(stat.st_mode):
                raise errors.ResourceBusy(rel_to)
            else:
                raise

    def delete(self, relpath):
        if urlutils.basename(relpath).startswith('.nfs'):
            raise errors.ResourceBusy(self.abspath(relpath))
        return self._decorated.delete(relpath)