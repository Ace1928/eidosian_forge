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
def _create_user_with_federated_objects(self, user, driver):
    if not user.get('federated'):
        if 'federated' in user:
            del user['federated']
        user = driver.create_user(user['id'], user)
        return user
    else:
        user_ref = user.copy()
        del user['federated']
        self._validate_federated_objects(user_ref['federated'])
        user = driver.create_user(user['id'], user)
        self._create_federated_objects(user_ref, user_ref['federated'])
        user['federated'] = user_ref['federated']
        return user