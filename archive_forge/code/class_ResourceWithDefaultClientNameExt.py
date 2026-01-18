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
class ResourceWithDefaultClientNameExt(resource.Resource):
    default_client_name = 'sample'
    required_service_extension = 'foo'
    properties_schema = {}