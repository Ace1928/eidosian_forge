from oslo_serialization import jsonutils
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from urllib import parse
class ZaqarQueue(resource.Resource):
    """A resource for managing Zaqar queues.

    Queue is a logical entity that groups messages. Ideally a queue is created
    per work type. For example, if you want to compress files, you would create
    a queue dedicated for this job. Any application that reads from this queue
    would only compress files.
    """
    default_client_name = 'zaqar'
    support_status = support.SupportStatus(version='2014.2')
    physical_resource_name_limit = 64
    PROPERTIES = NAME, METADATA = ('name', 'metadata')
    ATTRIBUTES = QUEUE_ID, HREF = ('queue_id', 'href')
    properties_schema = {NAME: properties.Schema(properties.Schema.STRING, _('Name of the queue instance to create.'), constraints=[constraints.Length(max=physical_resource_name_limit)]), METADATA: properties.Schema(properties.Schema.MAP, description=_('Arbitrary key/value metadata to store contextual information about this queue.'), update_allowed=True)}
    attributes_schema = {QUEUE_ID: attributes.Schema(_('ID of the queue.'), cache_mode=attributes.Schema.CACHE_NONE, support_status=support.SupportStatus(status=support.HIDDEN, version='6.0.0', previous_status=support.SupportStatus(status=support.DEPRECATED, message=_('Use get_resource|Ref command instead. For example: { get_resource : <resource_name> }'), version='2015.1', previous_status=support.SupportStatus(version='2014.1')))), HREF: attributes.Schema(_('The resource href of the queue.'))}

    def physical_resource_name(self):
        name = self.properties[self.NAME]
        if name is not None:
            return name
        return super(ZaqarQueue, self).physical_resource_name()

    def handle_create(self):
        """Create a zaqar message queue."""
        queue_name = self.physical_resource_name()
        queue = self.client().queue(queue_name, auto_create=False)
        metadata = self.properties.get('metadata')
        if metadata:
            queue.metadata(new_meta=metadata)
        self.resource_id_set(queue_name)

    def handle_update(self, json_snippet, tmpl_diff, prop_diff):
        """Update queue metadata."""
        if 'metadata' in prop_diff:
            queue = self.client().queue(self.resource_id, auto_create=False)
            metadata = prop_diff['metadata']
            queue.metadata(new_meta=metadata)

    def handle_delete(self):
        """Delete a zaqar message queue."""
        if not self.resource_id:
            return
        with self.client_plugin().ignore_not_found:
            self.client().queue(self.resource_id, auto_create=False).delete()

    def href(self):
        client = self.client()
        queue_name = self.physical_resource_name()
        return '%s/v%s/queues/%s' % (client.api_url.rstrip('/'), client.api_version, parse.quote(queue_name))

    def _resolve_attribute(self, name):
        if name == self.QUEUE_ID:
            return self.resource_id
        elif name == self.HREF:
            return self.href()

    def _show_resource(self):
        queue = self.client().queue(self.resource_id, auto_create=False)
        metadata = queue.metadata()
        return {self.METADATA: metadata}

    def parse_live_resource_data(self, resource_properties, resource_data):
        name = self.resource_id
        if name == super(ZaqarQueue, self).physical_resource_name():
            name = None
        return {self.NAME: name, self.METADATA: resource_data[self.METADATA]}