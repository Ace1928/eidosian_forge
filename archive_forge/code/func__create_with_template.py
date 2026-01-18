from requests import exceptions
from heat.common import exception
from heat.common.i18n import _
from heat.common import template_format
from heat.common import urlfetch
from heat.engine import attributes
from heat.engine import properties
from heat.engine.resources import stack_resource
def _create_with_template(self, resource_adopt_data=None):
    template = self.child_template()
    return self.create_with_template(template, self.child_params(), self.properties[self.TIMEOUT_IN_MINS], adopt_data=resource_adopt_data)