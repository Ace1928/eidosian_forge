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
class TemplatePluginFixture(fixtures.Fixture):

    def __init__(self, templates=None):
        templates = templates or {}
        super(TemplatePluginFixture, self).__init__()
        self.templates = [extension.Extension(k, None, v, None) for k, v in templates.items()]

    def _get_template_extension_manager(self):
        return extension.ExtensionManager.make_test_instance(self.templates)

    def setUp(self):
        super(TemplatePluginFixture, self).setUp()

        def clear_template_classes():
            template._template_classes = None
        clear_template_classes()
        self.useFixture(fixtures.MockPatchObject(template, '_get_template_extension_manager', new=self._get_template_extension_manager))
        self.addCleanup(clear_template_classes)