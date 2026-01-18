from ... import version_info  # noqa: F401
from ... import controldir, errors
from ... import transport as _mod_transport
class SmartHgProber(controldir.Prober):
    _supported_schemes = ['http', 'https']

    @classmethod
    def priority(klass, transport):
        if 'hg' in transport.base:
            return 90
        return 99

    @staticmethod
    def _has_hg_http_smart_server(transport, external_url):
        """Check if there is a Mercurial smart server at the remote location.

        :param transport: Transport to check
        :param externa_url: External URL for transport
        :return: Boolean indicating whether transport is backed onto hg
        """
        from breezy.urlutils import urlparse
        parsed_url = urlparse.urlparse(external_url)
        parsed_url = parsed_url._replace(query='cmd=capabilities')
        url = urlparse.urlunparse(parsed_url)
        resp = transport.request('GET', url, headers={'Accept': 'application/mercurial-0.1'})
        if resp.status == 404:
            return False
        if resp.status == 406:
            return False
        ct = resp.getheader('Content-Type')
        if ct is None:
            return False
        return ct.startswith('application/mercurial')

    @classmethod
    def probe_transport(klass, transport):
        try:
            external_url = transport.external_url()
        except errors.InProcessTransport:
            raise errors.NotBranchError(path=transport.base)
        scheme = external_url.split(':')[0]
        if scheme not in klass._supported_schemes:
            raise errors.NotBranchError(path=transport.base)
        from breezy import urlutils
        external_url = urlutils.strip_segment_parameters(external_url)
        if external_url.startswith('http:') or external_url.startswith('https:'):
            if klass._has_hg_http_smart_server(transport, external_url):
                return SmartHgDirFormat()
        raise errors.NotBranchError(path=transport.base)

    @classmethod
    def known_formats(cls):
        return [SmartHgDirFormat()]