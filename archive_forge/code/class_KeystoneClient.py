import collections
import uuid
import weakref
from keystoneauth1 import exceptions as ks_exception
from keystoneauth1.identity import generic as ks_auth
from keystoneclient.v3 import client as kc_v3
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import importutils
from heat.common import config
from heat.common import context
from heat.common import exception
from heat.common.i18n import _
from heat.common import password_gen
class KeystoneClient(object):
    """Keystone Auth Client.

    Delay choosing the backend client module until the client's class
    needs to be initialized.
    """

    def __new__(cls, context, region_name=None):
        if cfg.CONF.keystone_backend == _default_keystone_backend:
            return KsClientWrapper(context, region_name)
        else:
            return importutils.import_object(cfg.CONF.keystone_backend, context)