import datetime
import random as _random
import uuid
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from keystone.common import sql
from keystone.common import utils
from keystone import exception
from keystone.i18n import _
from keystone.oauth1.backends import base
def _get_request_token(self, session, request_token_id):
    token_ref = session.get(RequestToken, request_token_id)
    if token_ref is None:
        raise exception.NotFound(_('Request token not found'))
    return token_ref