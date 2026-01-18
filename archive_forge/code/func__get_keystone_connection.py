from collections import defaultdict
from collections import namedtuple
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import loading
from openstack import connection
from oslo_config import cfg
from oslo_log import log
from oslo_limit import exception
from oslo_limit import opts
def _get_keystone_connection():
    global _SDK_CONNECTION
    if not _SDK_CONNECTION:
        try:
            auth = loading.load_auth_from_conf_options(CONF, group='oslo_limit')
            session = loading.load_session_from_conf_options(CONF, group='oslo_limit', auth=auth)
            _SDK_CONNECTION = connection.Connection(session=session).identity
        except (ksa_exceptions.NoMatchingPlugin, ksa_exceptions.MissingRequiredOptions, ksa_exceptions.MissingAuthPlugin, ksa_exceptions.DiscoveryFailure, ksa_exceptions.Unauthorized) as e:
            msg = 'Unable to initialize OpenStackSDK session: %s' % e
            LOG.error(msg)
            raise exception.SessionInitError(e)
    return _SDK_CONNECTION