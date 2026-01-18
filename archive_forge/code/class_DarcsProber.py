from breezy import controldir, errors
from ... import version_info  # noqa: F401
class DarcsProber(controldir.Prober):

    @classmethod
    def priority(klass, transport):
        if 'darcs' in transport.base:
            return 90
        return 100

    @classmethod
    def probe_transport(klass, transport):
        if transport.has_any(['_darcs/format', '_darcs/inventory']):
            return DarcsDirFormat()
        raise errors.NotBranchError(path=transport.base)

    @classmethod
    def known_formats(cls):
        return [DarcsDirFormat()]