import logging
import keystoneauth1.access.service_catalog as sc
import keystoneauth1.identity.generic as auth_plugin
from keystoneauth1 import session as ks_session
import mistralclient.api.httpclient as api
from mistralclient import auth as mistral_auth
from oslo_serialization import jsonutils
@staticmethod
def _separate_target_reqs(req):
    """Separates parameters into target and non-target ones.

        target_* parameters are rekeyed by removing the prefix.

        :param req: Request dict containing the parameters for Keystone
        authentication.
        :return: list of [non-target, target] request parameters
        """
    r = {}
    target_r = {}
    target_prefix = 'target_'
    for key in req:
        if key.startswith(target_prefix):
            target_r[key[len(target_prefix):]] = req[key]
        else:
            r[key] = req[key]
    return [r, target_r]