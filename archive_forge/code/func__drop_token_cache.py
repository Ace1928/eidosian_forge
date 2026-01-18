import base64
import datetime
import uuid
from oslo_log import log
from oslo_utils import timeutils
from keystone.common import cache
from keystone.common import manager
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.federation import constants
from keystone.i18n import _
from keystone.models import token_model
from keystone import notifications
def _drop_token_cache(self, service, resource_type, operation, payload):
    """Invalidate the entire token cache.

        This is a handy private utility method that should be used when
        consuming notifications that signal invalidating the token cache.

        """
    if CONF.token.cache_on_issue or CONF.token.caching:
        TOKENS_REGION.invalidate()