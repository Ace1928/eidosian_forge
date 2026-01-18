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
class GenericResource(resource.Resource):
    """Dummy resource for use in tests."""
    properties_schema = {}
    attributes_schema = collections.OrderedDict([('foo', attributes.Schema('A generic attribute')), ('Foo', attributes.Schema('Another generic attribute'))])

    @classmethod
    def is_service_available(cls, context):
        return (True, None)

    def handle_create(self):
        LOG.warning('Creating generic resource (Type "%s")', self.type())

    def handle_update(self, json_snippet, tmpl_diff, prop_diff):
        LOG.warning('Updating generic resource (Type "%s")', self.type())

    def handle_delete(self):
        LOG.warning('Deleting generic resource (Type "%s")', self.type())

    def _resolve_attribute(self, name):
        return self.name

    def handle_suspend(self):
        LOG.warning('Suspending generic resource (Type "%s")', self.type())

    def handle_resume(self):
        LOG.warning('Resuming generic resource (Type "%s")', self.type())