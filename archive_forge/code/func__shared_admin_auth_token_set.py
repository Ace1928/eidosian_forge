import functools
import flask
from oslo_log import log
from oslo_policy import opts
from oslo_policy import policy as common_policy
from oslo_utils import strutils
from keystone.common import authorization
from keystone.common import context
from keystone.common import policies
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
@classmethod
def _shared_admin_auth_token_set(cls):
    ctx = cls._get_oslo_req_context()
    return getattr(ctx, 'is_admin', False)