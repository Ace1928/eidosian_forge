from functools import partial
from oslo_log import log
import stevedore
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.backends import resource_options as ro
def _lookup_trust(self, trust_info):
    trust_id = trust_info.get('id')
    if not trust_id:
        raise exception.ValidationError(attribute='trust_id', target='trust')
    trust = PROVIDERS.trust_api.get_trust(trust_id)
    return trust