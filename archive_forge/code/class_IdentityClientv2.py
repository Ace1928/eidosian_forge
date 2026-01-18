import logging
from keystoneclient.v2_0 import client as identity_client_v2
from osc_lib import utils
from openstackclient.i18n import _
class IdentityClientv2(identity_client_v2.Client):
    """Tweak the earlier client class to deal with some changes"""

    def __getattr__(self, name):
        if name == 'projects':
            return self.tenants
        else:
            raise AttributeError(name)