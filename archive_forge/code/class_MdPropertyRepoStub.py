from collections import abc
from unittest import mock
import hashlib
import os.path
import oslo_config.cfg
from oslo_policy import policy as common_policy
import glance.api.policy
from glance.common import exception
import glance.context
from glance.policies import base as base_policy
from glance.tests.unit import base
class MdPropertyRepoStub(object):

    def add(self, prop):
        return 'mdprop_add'

    def get(self, ns, prop_name):
        return 'mdprop_get'

    def list(self, *args, **kwargs):
        return ['mdprop_list']

    def save(self, prop):
        return 'mdprop_save'

    def remove(self, prop):
        return 'mdprop_remove'