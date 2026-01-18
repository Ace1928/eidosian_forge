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
class MultiStepResource(GenericResource):
    properties_schema = {'create_steps': properties.Schema(properties.Schema.INTEGER, default=2), 'update_steps': properties.Schema(properties.Schema.INTEGER, default=2, update_allowed=True), 'delete_steps': properties.Schema(properties.Schema.INTEGER, default=2, update_allowed=True)}

    def handle_create(self):
        super(MultiStepResource, self).handle_create()
        return [None] * self.properties['create_steps']

    def check_create_complete(self, cookie):
        cookie.pop()
        return not cookie

    def handle_update(self, json_snippet, tmpl_diff, prop_diff):
        super(MultiStepResource, self).handle_update(json_snippet, tmpl_diff, prop_diff)
        return [None] * self.properties['update_steps']

    def check_update_complete(self, cookie):
        cookie.pop()
        return not cookie

    def handle_delete(self):
        super(MultiStepResource, self).handle_delete()
        return [None] * self.properties['delete_steps']

    def check_delete_complete(self, cookie):
        cookie.pop()
        return not cookie