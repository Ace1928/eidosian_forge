from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
class GenericContainer(resource.Resource):
    """A resource for creating Barbican generic container.

    A generic container is used for any type of secret that a user
    may wish to aggregate. There are no restrictions on the amount
    of secrets that can be held within this container.
    """
    support_status = support.SupportStatus(version='6.0.0')
    default_client_name = 'barbican'
    entity = 'containers'
    PROPERTIES = NAME, SECRETS = ('name', 'secrets')
    ATTRIBUTES = STATUS, CONTAINER_REF, SECRET_REFS, CONSUMERS = ('status', 'container_ref', 'secret_refs', 'consumers')
    _SECRETS_PROPERTIES = NAME, REF = ('name', 'ref')
    properties_schema = {NAME: properties.Schema(properties.Schema.STRING, _('Human-readable name for the container.')), SECRETS: properties.Schema(properties.Schema.LIST, _('References to secrets that will be stored in container.'), schema=properties.Schema(properties.Schema.MAP, schema={NAME: properties.Schema(properties.Schema.STRING, _('Name of the secret.'), required=True), REF: properties.Schema(properties.Schema.STRING, _('Reference to the secret.'), required=True, constraints=[constraints.CustomConstraint('barbican.secret')])}))}
    attributes_schema = {STATUS: attributes.Schema(_('The status of the container.'), type=attributes.Schema.STRING), CONTAINER_REF: attributes.Schema(_('The URI to the container.'), type=attributes.Schema.STRING), SECRET_REFS: attributes.Schema(_('The URIs to secrets stored in container.'), type=attributes.Schema.MAP), CONSUMERS: attributes.Schema(_('The URIs to container consumers.'), type=attributes.Schema.LIST)}

    def get_refs(self):
        secrets = self.properties.get(self.SECRETS) or []
        return [secret['ref'] for secret in secrets]

    def validate(self):
        super(GenericContainer, self).validate()
        refs = self.get_refs()
        if len(refs) != len(set(refs)):
            msg = _('Duplicate refs are not allowed.')
            raise exception.StackValidationFailed(message=msg)

    def create_container(self):
        if self.properties[self.SECRETS]:
            secrets = dict(((secret['name'], secret['ref']) for secret in self.properties[self.SECRETS]))
        else:
            secrets = {}
        info = {'secret_refs': secrets}
        if self.properties[self.NAME] is not None:
            info.update({'name': self.properties[self.NAME]})
        return self.client_plugin().create_generic_container(**info)

    def handle_create(self):
        container_ref = self.create_container().store()
        self.resource_id_set(container_ref)
        return container_ref

    def check_create_complete(self, container_href):
        container = self.client().containers.get(container_href)
        if container.status == 'ERROR':
            reason = container.error_reason
            code = container.error_status_code
            msg = _("Container '%(name)s' creation failed: %(code)s - %(reason)s") % {'name': self.name, 'code': code, 'reason': reason}
            raise exception.ResourceInError(status_reason=msg, resource_status=container.status)
        return container.status == 'ACTIVE'

    def _resolve_attribute(self, name):
        if self.resource_id is None:
            return
        container = self.client().containers.get(self.resource_id)
        return getattr(container, name, None)