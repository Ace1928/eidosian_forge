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
class ResWithStringPropAndAttr(GenericResource):
    properties_schema = {'a_string': properties.Schema(properties.Schema.STRING)}
    attributes_schema = {'string': attributes.Schema('A string')}

    def _resolve_attribute(self, name):
        try:
            return self.properties['a_%s' % name]
        except KeyError:
            return None