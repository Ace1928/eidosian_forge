import datetime as dt
import json
from unittest import mock
import uuid
from oslo_utils import timeutils
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils as heat_timeutils
from heat.db import api as db_api
from heat.db import models
from heat.engine import api
from heat.engine.cfn import parameters as cfn_param
from heat.engine import event
from heat.engine import parent_rsrc
from heat.engine import stack as parser
from heat.engine import template
from heat.objects import event as event_object
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import utils
class TranslateFilterTest(common.HeatTestCase):
    scenarios = [('single+single', dict(inputs={'stack_status': 'COMPLETE', 'status': 'FAILED'}, expected={'status': ['COMPLETE', 'FAILED']})), ('none+single', dict(inputs={'name': 'n1'}, expected={'name': 'n1'})), ('single+none', dict(inputs={'stack_name': 'n1'}, expected={'name': 'n1'})), ('none+list', dict(inputs={'action': ['a1', 'a2']}, expected={'action': ['a1', 'a2']})), ('list+none', dict(inputs={'stack_action': ['a1', 'a2']}, expected={'action': ['a1', 'a2']})), ('single+list', dict(inputs={'stack_owner': 'u1', 'username': ['u2', 'u3']}, expected={'username': ['u1', 'u2', 'u3']})), ('list+single', dict(inputs={'parent': ['s1', 's2'], 'owner_id': 's3'}, expected={'owner_id': ['s1', 's2', 's3']})), ('list+list', dict(inputs={'stack_name': ['n1', 'n2'], 'name': ['n3', 'n4']}, expected={'name': ['n1', 'n2', 'n3', 'n4']})), ('full_status_split', dict(inputs={'stack_status': 'CREATE_COMPLETE'}, expected={'action': 'CREATE', 'status': 'COMPLETE'})), ('full_status_split_merge', dict(inputs={'stack_status': 'CREATE_COMPLETE', 'status': 'CREATE_FAILED'}, expected={'action': 'CREATE', 'status': ['COMPLETE', 'FAILED']})), ('action_status_merge', dict(inputs={'action': ['UPDATE', 'CREATE'], 'status': 'CREATE_FAILED'}, expected={'action': ['CREATE', 'UPDATE'], 'status': 'FAILED'}))]

    def test_stack_filter_translate(self):
        actual = api.translate_filters(self.inputs)
        self.assertEqual(self.expected, actual)