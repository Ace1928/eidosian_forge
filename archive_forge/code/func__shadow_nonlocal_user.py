import copy
import functools
import itertools
import operator
import os
import threading
import uuid
from oslo_config import cfg
from oslo_log import log
from pycadf import reason
from keystone import assignment  # TODO(lbragstad): Decouple this dependency
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.mapping_backends import mapping
from keystone import notifications
from oslo_utils import timeutils
@MEMOIZE
def _shadow_nonlocal_user(self, user):
    try:
        return PROVIDERS.shadow_users_api.get_user(user['id'])
    except exception.UserNotFound:
        return PROVIDERS.shadow_users_api.create_nonlocal_user(user)