from ... import version_info  # noqa: F401
from ... import controldir, errors
from ...transport import register_transport_proto
class CVSProber(controldir.Prober):

    @classmethod
    def priority(klass, transport):
        return 100

    @classmethod
    def probe_transport(klass, transport):
        if transport.has('CVS') and transport.has('CVS/Repository'):
            return CVSDirFormat()
        raise errors.NotBranchError(path=transport.base)

    @classmethod
    def known_formats(cls):
        return [CVSDirFormat()]