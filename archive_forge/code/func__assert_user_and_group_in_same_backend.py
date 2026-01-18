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
def _assert_user_and_group_in_same_backend(self, user_entity_id, user_driver, group_entity_id, group_driver):
    """Ensure that user and group IDs are backed by the same backend.

        Raise a CrossBackendNotAllowed exception if they are not from the same
        backend, otherwise return None.

        """
    if user_driver is not group_driver:
        user_driver.get_user(user_entity_id)
        group_driver.get_group(group_entity_id)
        raise exception.CrossBackendNotAllowed(group_id=group_entity_id, user_id=user_entity_id)