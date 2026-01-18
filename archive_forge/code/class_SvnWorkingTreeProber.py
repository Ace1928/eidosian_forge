from ... import version_info  # noqa: F401
from ... import controldir, errors
from ... import transport as _mod_transport
from ...revisionspec import revspec_registry
class SvnWorkingTreeProber(controldir.Prober):

    @classmethod
    def priority(klass, transport):
        return 100

    def probe_transport(self, transport):
        try:
            transport.local_abspath('.')
        except errors.NotLocalUrl:
            raise errors.NotBranchError(path=transport.base)
        else:
            if transport.has('.svn'):
                return SvnWorkingTreeDirFormat()
            raise errors.NotBranchError(path=transport.base)

    @classmethod
    def known_formats(cls):
        return [SvnWorkingTreeDirFormat()]