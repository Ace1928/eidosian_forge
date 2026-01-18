import datetime
from unittest import mock
import uuid
from oslo_serialization import jsonutils as json
from oslo_utils import timeutils
from heat.common import identifier
from heat.common import template_format
from heat.engine.clients.os import heat_plugin
from heat.engine.clients.os import swift as swift_plugin
from heat.engine import environment
from heat.engine import resource
from heat.engine.resources.openstack.heat import wait_condition_handle as h_wch
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.objects import resource as resource_objects
from heat.tests import common
from heat.tests import utils
class mock_swift(object):

    @staticmethod
    def put_container(container, **kwargs):
        pass

    @staticmethod
    def put_object(container, object, contents, **kwargs):
        pass