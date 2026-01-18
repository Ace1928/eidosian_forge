import copy
import hashlib
import json
import fixtures
from stevedore import extension
from heat.common import exception
from heat.common import template_format
from heat.engine.cfn import functions as cfn_funcs
from heat.engine.cfn import parameters as cfn_p
from heat.engine.cfn import template as cfn_t
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine import function
from heat.engine.hot import template as hot_t
from heat.engine import node_data
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import stk_defn
from heat.engine import template
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
class TestTemplatePluginManager(common.HeatTestCase):

    def test_template_NEW_good(self):

        class NewTemplate(template.Template):
            SECTIONS = VERSION, MAPPINGS, CONDITIONS, PARAMETERS = ('NEWTemplateFormatVersion', '__undefined__', 'conditions', 'parameters')
            RESOURCES = 'thingies'

            def param_schemata(self, param_defaults=None):
                pass

            def get_section_name(self, section):
                pass

            def parameters(self, stack_identifier, user_params, param_defaults=None):
                pass

            def resource_definitions(self, stack):
                pass

            def add_resource(self, definition, name=None):
                pass

            def outputs(self, stack):
                pass

            def __getitem__(self, section):
                return {}

        class NewTemplatePrint(function.Function):

            def result(self):
                return 'always this'
        self.useFixture(TemplatePluginFixture({'NEWTemplateFormatVersion.2345-01-01': NewTemplate}))
        t = {'NEWTemplateFormatVersion': '2345-01-01'}
        tmpl = template.Template(t)
        err = tmpl.validate()
        self.assertIsNone(err)