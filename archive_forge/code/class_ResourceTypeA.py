import copy
from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.db import api as db_api
from heat.engine.clients.os.keystone import fake_keystoneclient
from heat.engine import environment
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import service
from heat.engine import stack
from heat.engine import support
from heat.engine import template
from heat.objects import stack as stack_object
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
class ResourceTypeA(ResourceTypeB):
    support_status = support.SupportStatus(status=support.DEPRECATED, message='deprecation_msg', version='2014.2', substitute_class=ResourceTypeB)
    count_a = 0

    def update(self, after, before=None, prev_resource=None):
        ResourceTypeA.count_a += 1
        yield