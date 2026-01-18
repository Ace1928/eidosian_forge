from collections import namedtuple
from . import compat
from . import exceptions
from . import misc
from . import normalizers
from . import uri
class ParseResultMixin(object):

    def _generate_authority(self, attributes):
        userinfo, host, port = (attributes[p] for p in ('userinfo', 'host', 'port'))
        if self.userinfo != userinfo or self.host != host or self.port != port:
            if port:
                port = '{0}'.format(port)
            return normalizers.normalize_authority((compat.to_str(userinfo, self.encoding), compat.to_str(host, self.encoding), port))
        if isinstance(self.authority, bytes):
            return self.authority.decode('utf-8')
        return self.authority

    def geturl(self):
        """Shim to match the standard library method."""
        return self.unsplit()

    @property
    def hostname(self):
        """Shim to match the standard library."""
        return self.host

    @property
    def netloc(self):
        """Shim to match the standard library."""
        return self.authority

    @property
    def params(self):
        """Shim to match the standard library."""
        return self.query