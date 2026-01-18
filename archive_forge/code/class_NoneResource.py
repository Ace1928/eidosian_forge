import uuid
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
class NoneResource(resource.Resource):
    """Enables easily disabling certain resources via the resource_registry.

    It does nothing, but can effectively stub out any other resource because it
    will accept any properties and return any attribute (as None). Note this
    resource always does nothing on update (e.g it is not replaced even if a
    change to the stubbed resource properties would cause replacement).
    """
    support_status = support.SupportStatus(version='5.0.0')
    properties_schema = {}
    attributes_schema = {}
    IS_PLACEHOLDER = 'is_placeholder'

    def _needs_update(self, after, before, after_props, before_props, prev_resource, check_init_complete=True):
        return False

    def frozen_definition(self):
        return self.t.freeze(properties=properties.Properties(schema={}, data={}))

    def reparse(self, client_resolve=True):
        self.properties = properties.Properties(schema={}, data={})
        self.translate_properties(self.properties, client_resolve)

    def handle_create(self):
        self.resource_id_set(str(uuid.uuid4()))
        self.data_set(self.IS_PLACEHOLDER, 'True')

    def validate(self):
        pass

    def get_attribute(self, key, *path):
        return None

    def handle_delete(self):
        if not self.data().get(self.IS_PLACEHOLDER):
            return super(NoneResource, self).handle_delete()
        return self.resource_id