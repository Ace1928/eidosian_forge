from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from urllib import parse
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients.os import swift
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
class SwiftSignalHandle(resource.Resource):
    """Resource for managing signals from Swift resources.

    This resource is same as WaitConditionHandle, but designed for using by
    Swift resources.
    """
    support_status = support.SupportStatus(version='2014.2')
    default_client_name = 'swift'
    properties_schema = {}
    ATTRIBUTES = TOKEN, ENDPOINT, CURL_CLI = ('token', 'endpoint', 'curl_cli')
    attributes_schema = {TOKEN: attributes.Schema(_('Tokens are not needed for Swift TempURLs. This attribute is being kept for compatibility with the OS::Heat::WaitConditionHandle resource.'), cache_mode=attributes.Schema.CACHE_NONE, type=attributes.Schema.STRING), ENDPOINT: attributes.Schema(_('Endpoint/url which can be used for signalling handle.'), cache_mode=attributes.Schema.CACHE_NONE, type=attributes.Schema.STRING), CURL_CLI: attributes.Schema(_('Convenience attribute, provides curl CLI command prefix, which can be used for signalling handle completion or failure. You can signal success by adding --data-binary \'{"status": "SUCCESS"}\' , or signal failure by adding --data-binary \'{"status": "FAILURE"}\'.'), cache_mode=attributes.Schema.CACHE_NONE, type=attributes.Schema.STRING)}

    def handle_create(self):
        cplugin = self.client_plugin()
        url = cplugin.get_signal_url(self.stack.id, self.physical_resource_name())
        self.data_set(self.ENDPOINT, url)
        self.resource_id_set(self.physical_resource_name())

    def _resolve_attribute(self, key):
        if self.resource_id:
            if key == self.TOKEN:
                return ''
            elif key == self.ENDPOINT:
                return self.data().get(self.ENDPOINT)
            elif key == self.CURL_CLI:
                return "curl -i -X PUT '%s'" % self.data().get(self.ENDPOINT)

    def handle_delete(self):
        cplugin = self.client_plugin()
        client = cplugin.client()
        while True:
            try:
                client.delete_object(self.stack.id, self.physical_resource_name())
            except Exception as exc:
                cplugin.ignore_not_found(exc)
                break
        try:
            client.delete_container(self.stack.id)
        except Exception as exc:
            if cplugin.is_not_found(exc) or cplugin.is_conflict(exc):
                pass
            else:
                raise
        self.data_delete(self.ENDPOINT)

    def get_reference_id(self):
        return self.data().get(self.ENDPOINT)