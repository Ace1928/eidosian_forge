import base64
import collections
from urllib import parse
from novaclient import api_versions
from novaclient import base
from novaclient import crypto
from novaclient import exceptions
from novaclient.i18n import _
def _validate_create_nics(self, nics):
    if self.api_version > api_versions.APIVersion('2.36'):
        if not nics:
            raise ValueError('nics are required after microversion 2.36')
    elif nics and (not isinstance(nics, (list, tuple))):
        raise ValueError('nics must be a list or a tuple, not %s' % type(nics))