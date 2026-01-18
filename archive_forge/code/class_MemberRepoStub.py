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
class MemberRepoStub(object):
    image = None

    def add(self, image_member):
        image_member.output = 'member_repo_add'

    def get(self, *args, **kwargs):
        return 'member_repo_get'

    def save(self, image_member, from_state=None):
        image_member.output = 'member_repo_save'

    def list(self, *args, **kwargs):
        return 'member_repo_list'

    def remove(self, image_member):
        image_member.output = 'member_repo_remove'