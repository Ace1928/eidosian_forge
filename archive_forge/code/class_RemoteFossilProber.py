from ... import version_info  # noqa: F401
from ... import controldir, errors
class RemoteFossilProber(controldir.Prober):

    @classmethod
    def priority(klass, transport):
        return 95

    @classmethod
    def probe_transport(klass, transport):
        from breezy.transport.http.urllib import HttpTransport
        if not isinstance(transport, HttpTransport):
            raise errors.NotBranchError(path=transport.base)
        response = transport.request('POST', transport.base, headers={'Content-Type': 'application/x-fossil'})
        if response.status == 501:
            raise errors.NotBranchError(path=transport.base)
        ct = response.getheader('Content-Type')
        if ct is None:
            raise errors.NotBranchError(path=transport.base)
        if ct.split(';')[0] != 'application/x-fossil':
            raise errors.NotBranchError(path=transport.base)
        return FossilDirFormat()

    @classmethod
    def known_formats(cls):
        return [FossilDirFormat()]