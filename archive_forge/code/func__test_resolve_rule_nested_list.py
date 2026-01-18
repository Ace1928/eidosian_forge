import copy
from unittest import mock
from heat.common import exception
from heat.engine.cfn import functions as cfn_funcs
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
from heat.engine import parameters
from heat.engine import properties
from heat.engine import translation
from heat.tests import common
def _test_resolve_rule_nested_list(self):

    class FakeClientPlugin(object):

        def find_name_id(self, entity=None, value=None):
            if entity == 'net':
                return 'net1_id'
            if entity == 'port':
                return 'port1_id'
    schema = {'instances': properties.Schema(properties.Schema.LIST, schema=properties.Schema(properties.Schema.MAP, schema={'networks': properties.Schema(properties.Schema.LIST, schema=properties.Schema(properties.Schema.MAP, schema={'port': properties.Schema(properties.Schema.STRING), 'net': properties.Schema(properties.Schema.STRING)}))}))}
    return (FakeClientPlugin(), schema)