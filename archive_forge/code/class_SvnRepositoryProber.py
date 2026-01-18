from ... import version_info  # noqa: F401
from ... import controldir, errors
from ... import transport as _mod_transport
from ...revisionspec import revspec_registry
class SvnRepositoryProber(controldir.Prober):
    _supported_schemes = ['http', 'https', 'file', 'svn']

    @classmethod
    def priority(klass, transport):
        if 'svn' in transport.base:
            return 90
        return 100

    def probe_transport(self, transport):
        try:
            url = transport.external_url()
        except errors.InProcessTransport:
            raise errors.NotBranchError(path=transport.base)
        scheme = url.split(':')[0]
        if scheme.startswith('svn+') or scheme == 'svn':
            raise SubversionUnsupportedError()
        if scheme not in self._supported_schemes:
            raise errors.NotBranchError(path=transport.base)
        if scheme == 'file':
            maybe = False
            subtransport = transport
            while subtransport:
                try:
                    if all([subtransport.has(name) for name in ['format', 'db', 'conf']]):
                        maybe = True
                        break
                except UnicodeEncodeError:
                    pass
                prevsubtransport = subtransport
                subtransport = prevsubtransport.clone('..')
                if subtransport.base == prevsubtransport.base:
                    break
            if not maybe:
                raise errors.NotBranchError(path=transport.base)
        if scheme in ('http', 'https'):
            priv_transport = getattr(transport, '_decorated', transport)
            try:
                headers = priv_transport._options('.')
            except (errors.InProcessTransport, _mod_transport.NoSuchFile, errors.InvalidHttpResponse):
                raise errors.NotBranchError(path=transport.base)
            else:
                dav_entries = set()
                for key, value in headers:
                    if key.upper() == 'DAV':
                        dav_entries.update([x.strip() for x in value.split(',')])
                if 'version-control' not in dav_entries:
                    raise errors.NotBranchError(path=transport.base)
        return SvnRepositoryFormat()

    @classmethod
    def known_formats(cls):
        return [SvnRepositoryFormat()]