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
class ResourceWithMultipleRequiredProps(GenericResource):
    properties_schema = {'Foo1': properties.Schema(properties.Schema.STRING, required=True), 'Foo2': properties.Schema(properties.Schema.STRING, required=True), 'Foo3': properties.Schema(properties.Schema.STRING, required=True)}