import copy
from unittest import mock
from oslo_config import cfg
from zunclient import exceptions as zc_exc
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import zun
from heat.engine.resources.openstack.zun import container
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def _assert_addresses(self, expected, actual):
    matched = True
    if len(expected) != len(actual):
        matched = False
    for key in expected:
        if key not in actual:
            matched = False
            break
        list1 = expected[key]
        list1 = sorted(list1, key=lambda x: sorted(x.values()))
        list2 = actual[key]
        list2 = sorted(list2, key=lambda x: sorted(x.values()))
        if list1 != list2:
            matched = False
            break
    if not matched:
        raise AssertionError('Addresses is unmatched:\n reference = ' + str(expected) + '\nactual = ' + str(actual))