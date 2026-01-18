import tempfile
from tempest.lib import exceptions
from novaclient.tests.functional import base
from novaclient.tests.functional.v2 import fake_crypto
def _serialize_kwargs(self, kwargs):
    kwargs_pairs = ['--%(key)s %(val)s' % {'key': key.replace('_', '-'), 'val': val} for key, val in kwargs.items()]
    return ' '.join(kwargs_pairs)