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
def _handle_shadow_and_local_users(self, driver, hints):
    federated_attributes = {'idp_id', 'protocol_id', 'unique_id'}
    fed_res = []
    for filter_ in hints.filters:
        if filter_['name'] in federated_attributes:
            return PROVIDERS.shadow_users_api.get_federated_users(hints)
        if filter_['name'] == 'name':
            fed_hints = copy.deepcopy(hints)
            fed_res = PROVIDERS.shadow_users_api.get_federated_users(fed_hints)
            break
    return driver.list_users(hints) + fed_res