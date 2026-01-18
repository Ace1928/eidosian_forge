import collections
from oslo_log import log as logging
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources import signal_responder
from heat.engine.resources import stack_resource
from heat.engine.resources import stack_user
from heat.engine import support
class StackResourceType(stack_resource.StackResource, GenericResource):

    def physical_resource_name(self):
        return 'cb2f2b28-a663-4683-802c-4b40c916e1ff'

    def set_template(self, nested_template, params):
        self.nested_template = nested_template
        self.nested_params = params

    def handle_create(self):
        return self.create_with_template(self.nested_template, self.nested_params)

    def handle_adopt(self, resource_data):
        return self.create_with_template(self.nested_template, self.nested_params, adopt_data=resource_data)

    def handle_delete(self):
        self.delete_nested()

    def has_nested(self):
        if self.nested() is not None:
            return True
        return False