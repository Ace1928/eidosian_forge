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
class ResourceWithAttributeType(GenericResource):
    attributes_schema = {'attr1': attributes.Schema('A generic attribute', type=attributes.Schema.STRING), 'attr2': attributes.Schema('Another generic attribute', type=attributes.Schema.MAP)}

    def _resolve_attribute(self, name):
        if name == 'attr1':
            return 'valid_sting'
        elif name == 'attr2':
            return 'invalid_type'