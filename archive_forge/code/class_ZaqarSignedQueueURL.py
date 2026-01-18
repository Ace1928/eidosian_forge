from oslo_serialization import jsonutils
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from urllib import parse
class ZaqarSignedQueueURL(resource.Resource):
    """A resource for managing signed URLs of Zaqar queues.

    Signed URLs allow to give specific access to queues, for example to be used
    as alarm notifications. To supply a signed queue URL to Aodh as an action
    URL, pass "zaqar://?" followed by the query_str attribute of the signed
    queue URL resource.
    """
    default_client_name = 'zaqar'
    support_status = support.SupportStatus(version='8.0.0')
    PROPERTIES = QUEUE, PATHS, TTL, METHODS = ('queue', 'paths', 'ttl', 'methods')
    ATTRIBUTES = SIGNATURE, EXPIRES, PATHS_ATTR, METHODS_ATTR, PROJECT, QUERY_STR = ('signature', 'expires', 'paths', 'methods', 'project', 'query_str')
    properties_schema = {QUEUE: properties.Schema(properties.Schema.STRING, _('Name of the queue instance to create a URL for.'), required=True), PATHS: properties.Schema(properties.Schema.LIST, description=_('List of allowed paths to be accessed. Default to allow queue messages URL.')), TTL: properties.Schema(properties.Schema.INTEGER, description=_('Time validity of the URL, in seconds. Default to one day.')), METHODS: properties.Schema(properties.Schema.LIST, description=_('List of allowed HTTP methods to be used. Default to allow GET.'), schema=properties.Schema(properties.Schema.STRING, constraints=[constraints.AllowedValues(['GET', 'DELETE', 'PATCH', 'POST', 'PUT'])]))}
    attributes_schema = {SIGNATURE: attributes.Schema(_('Signature of the URL built by Zaqar.')), EXPIRES: attributes.Schema(_('Expiration date of the URL.')), PATHS_ATTR: attributes.Schema(_('Comma-delimited list of paths for convenience.')), METHODS_ATTR: attributes.Schema(_('Comma-delimited list of methods for convenience.')), PROJECT: attributes.Schema(_('The ID of the Keystone project containing the queue.')), QUERY_STR: attributes.Schema(_('An HTTP URI query fragment.'))}

    def handle_create(self):
        queue = self.client().queue(self.properties[self.QUEUE])
        signed_url = queue.signed_url(paths=self.properties[self.PATHS], methods=self.properties[self.METHODS], ttl_seconds=self.properties[self.TTL])
        self.data_set(self.SIGNATURE, signed_url['signature'])
        self.data_set(self.EXPIRES, signed_url['expires'])
        self.data_set(self.PATHS_ATTR, jsonutils.dumps(signed_url['paths']))
        self.data_set(self.METHODS_ATTR, jsonutils.dumps(signed_url['methods']))
        self.data_set(self.PROJECT, signed_url['project'])
        self.resource_id_set(self.physical_resource_name())

    def _query_str(self, data):
        """Return the query fragment of a signed URI.

        This can be used, for example, for alarming.
        """
        paths = jsonutils.loads(data[self.PATHS_ATTR])
        methods = jsonutils.loads(data[self.METHODS_ATTR])
        query = {'signature': data[self.SIGNATURE], 'expires': data[self.EXPIRES], 'paths': ','.join(paths), 'methods': ','.join(methods), 'project_id': data[self.PROJECT], 'queue_name': self.properties[self.QUEUE]}
        return parse.urlencode(query)

    def handle_delete(self):
        return

    def _resolve_attribute(self, name):
        if not self.resource_id:
            return
        if name in (self.SIGNATURE, self.EXPIRES, self.PROJECT):
            return self.data()[name]
        elif name in (self.PATHS_ATTR, self.METHODS_ATTR):
            return jsonutils.loads(self.data()[name])
        elif name == self.QUERY_STR:
            return self._query_str(self.data())