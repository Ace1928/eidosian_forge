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
def _create_federated_objects(self, user_ref, fed_obj_list):
    for fed_obj in fed_obj_list:
        for protocols in fed_obj['protocols']:
            federated_dict = {'user_id': user_ref['id'], 'idp_id': fed_obj['idp_id'], 'protocol_id': protocols['protocol_id'], 'unique_id': protocols['unique_id'], 'display_name': user_ref['name']}
            self.shadow_users_api.create_federated_object(federated_dict)