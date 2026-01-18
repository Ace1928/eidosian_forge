import datetime
from unittest import mock
import glance_store
from oslo_config import cfg
import oslo_messaging
import webob
import glance.async_
from glance.common import exception
from glance.common import timeutils
import glance.context
from glance import notifier
import glance.tests.unit.utils as unit_test_utils
from glance.tests import utils
class ImageMemberRepoStub(object):

    def remove(self, *args, **kwargs):
        return 'image_member_from_remove'

    def save(self, *args, **kwargs):
        return 'image_member_from_save'

    def add(self, *args, **kwargs):
        return 'image_member_from_add'

    def get(self, *args, **kwargs):
        return 'image_member_from_get'

    def list(self, *args, **kwargs):
        return ['image_members_from_list']