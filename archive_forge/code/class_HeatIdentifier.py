import collections
import re
from oslo_utils import encodeutils
from urllib import parse as urlparse
from heat.common.i18n import _
class HeatIdentifier(collections.abc.Mapping):
    FIELDS = TENANT, STACK_NAME, STACK_ID, PATH = ('tenant', 'stack_name', 'stack_id', 'path')
    path_re = re.compile('stacks/([^/]+)/([^/]+)(.*)')

    def __init__(self, tenant, stack_name, stack_id, path=''):
        """Initialise a HeatIdentifier.

        Identifier is initialized from a Tenant ID, Stack name, Stack ID
        and optional path. If a path is supplied and it does not begin with
        "/", a "/" will be prepended.
        """
        if path and (not path.startswith('/')):
            path = '/' + path
        if '/' in stack_name:
            raise ValueError(_('Stack name may not contain "/"'))
        self.identity = {self.TENANT: tenant, self.STACK_NAME: stack_name, self.STACK_ID: str(stack_id), self.PATH: path}

    @classmethod
    def from_arn(cls, arn):
        """Generate a new HeatIdentifier by parsing the supplied ARN."""
        fields = arn.split(':')
        if len(fields) < 6 or fields[0].lower() != 'arn':
            raise ValueError(_('"%s" is not a valid ARN') % arn)
        id_fragment = ':'.join(fields[5:])
        path = cls.path_re.match(id_fragment)
        if fields[1] != 'openstack' or fields[2] != 'heat' or (not path):
            raise ValueError(_('"%s" is not a valid Heat ARN') % arn)
        return cls(urlparse.unquote(fields[4]), urlparse.unquote(path.group(1)), urlparse.unquote(path.group(2)), urlparse.unquote(path.group(3)))

    @classmethod
    def from_arn_url(cls, url):
        """Generate a new HeatIdentifier by parsing the supplied URL.

        The URL is expected to contain a valid arn as part of the path.
        """
        urlp = urlparse.urlparse(url)
        if urlp.scheme not in ('http', 'https') or not urlp.netloc or (not urlp.path):
            raise ValueError(_('"%s" is not a valid URL') % url)
        arn_url_prefix = '/arn%3Aopenstack%3Aheat%3A%3A'
        match = re.search(arn_url_prefix, urlp.path, re.IGNORECASE)
        if match is None:
            raise ValueError(_('"%s" is not a valid ARN URL') % url)
        url_arn = urlp.path[match.start() + 1:]
        arn = urlparse.unquote(url_arn)
        return cls.from_arn(arn)

    def arn(self):
        """Return as an ARN.

        Returned in the form:
            arn:openstack:heat::<tenant>:stacks/<stack_name>/<stack_id><path>
        """
        return 'arn:openstack:heat::%s:%s' % (urlparse.quote(self.tenant, ''), self._tenant_path())

    def arn_url_path(self):
        """Return an ARN quoted correctly for use in a URL."""
        return '/' + urlparse.quote(self.arn())

    def url_path(self):
        """Return a URL-encoded path segment of a URL.

        Returned in the form:
            <tenant>/stacks/<stack_name>/<stack_id><path>
        """
        return '/'.join((urlparse.quote(self.tenant, ''), self._tenant_path()))

    def _tenant_path(self):
        """URL-encoded path segment of a URL within a particular tenant.

        Returned in the form:
            stacks/<stack_name>/<stack_id><path>
        """
        return 'stacks/%s%s' % (self.stack_path(), urlparse.quote(encodeutils.safe_encode(self.path)))

    def stack_path(self):
        """Return a URL-encoded path segment of a URL without a tenant.

        Returned in the form:
            <stack_name>/<stack_id>
        """
        return '%s/%s' % (urlparse.quote(self.stack_name, ''), urlparse.quote(self.stack_id, ''))

    def _path_components(self):
        """Return a list of the path components."""
        return self.path.lstrip('/').split('/')

    def __getattr__(self, attr):
        """Return a component of the identity when accessed as an attribute."""
        if attr not in self.FIELDS:
            raise AttributeError(_('Unknown attribute "%s"') % attr)
        return self.identity[attr]

    def __getitem__(self, key):
        """Return one of the components of the identity."""
        if key not in self.FIELDS:
            raise KeyError(_('Unknown attribute "%s"') % key)
        return self.identity[key]

    def __len__(self):
        """Return the number of components in an identity."""
        return len(self.FIELDS)

    def __contains__(self, key):
        return key in self.FIELDS

    def __iter__(self):
        return iter(self.FIELDS)

    def __repr__(self):
        return repr(dict(self))