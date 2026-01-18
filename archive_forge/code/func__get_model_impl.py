from collections import defaultdict
from collections import namedtuple
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import loading
from openstack import connection
from oslo_config import cfg
from oslo_log import log
from oslo_limit import exception
from oslo_limit import opts
def _get_model_impl(self, usage_callback, cache=True):
    """get the enforcement model based on configured model in keystone."""
    model = self._get_enforcement_model()
    for impl in _MODELS:
        if model == impl.name:
            return impl(usage_callback, cache=cache)
    raise ValueError('enforcement model %s is not supported' % model)