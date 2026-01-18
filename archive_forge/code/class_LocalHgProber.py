from ... import version_info  # noqa: F401
from ... import controldir, errors
from ... import transport as _mod_transport
class LocalHgProber(controldir.Prober):

    @classmethod
    def priority(klass, transport):
        return 100

    @staticmethod
    def _has_hg_dumb_repository(transport):
        try:
            return transport.has_any(['.hg/requires', '.hg/00changelog.i'])
        except (_mod_transport.NoSuchFile, errors.PermissionDenied, errors.InvalidHttpResponse):
            return False

    @classmethod
    def probe_transport(klass, transport):
        """Our format is present if the transport has a '.hg/' subdir."""
        if klass._has_hg_dumb_repository(transport):
            return LocalHgDirFormat()
        raise errors.NotBranchError(path=transport.base)

    @classmethod
    def known_formats(cls):
        return [LocalHgDirFormat()]