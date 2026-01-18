from oslo_utils import encodeutils
import urllib.parse
import warlock
from glanceclient.common import utils
from glanceclient.v2 import schemas
class PropertyController(object):

    def __init__(self, http_client, schema_client):
        self.http_client = http_client
        self.schema_client = schema_client

    @utils.memoized_property
    def model(self):
        schema = self.schema_client.get('metadefs/property')
        return warlock.model_factory(schema.raw(), base_class=schemas.SchemaBasedModel)

    @utils.add_req_id_to_object()
    def create(self, namespace, **kwargs):
        """Create a property.

        :param namespace: Name of a namespace the property will belong.
        :param kwargs: Unpacked property object.
        """
        try:
            prop = self.model(kwargs)
        except (warlock.InvalidOperation, ValueError) as e:
            raise TypeError(encodeutils.exception_to_unicode(e))
        url = '/v2/metadefs/namespaces/%(namespace)s/properties' % {'namespace': namespace}
        resp, body = self.http_client.post(url, data=prop)
        body.pop('self', None)
        return (self.model(**body), resp)

    def update(self, namespace, prop_name, **kwargs):
        """Update a property.

        :param namespace: Name of a namespace the property belongs.
        :param prop_name: Name of a property (old one).
        :param kwargs: Unpacked property object.
        """
        prop = self.get(namespace, prop_name)
        for key, value in kwargs.items():
            try:
                setattr(prop, key, value)
            except warlock.InvalidOperation as e:
                raise TypeError(encodeutils.exception_to_unicode(e))
        url = '/v2/metadefs/namespaces/%(namespace)s/properties/%(prop_name)s' % {'namespace': namespace, 'prop_name': prop_name}
        resp, _ = self.http_client.put(url, data=prop.wrapped)
        req_id_hdr = {'x-openstack-request-id': utils._extract_request_id(resp)}
        return self._get(namespace, prop.name, req_id_hdr)

    def get(self, namespace, prop_name):
        return self._get(namespace, prop_name)

    @utils.add_req_id_to_object()
    def _get(self, namespace, prop_name, header=None):
        url = '/v2/metadefs/namespaces/%(namespace)s/properties/%(prop_name)s' % {'namespace': namespace, 'prop_name': prop_name}
        header = header or {}
        resp, body = self.http_client.get(url, headers=header)
        body.pop('self', None)
        body['name'] = prop_name
        return (self.model(**body), resp)

    @utils.add_req_id_to_generator()
    def list(self, namespace, **kwargs):
        """Retrieve a listing of metadata properties.

        :returns: generator over list of objects
        """
        url = '/v2/metadefs/namespaces/%(namespace)s/properties' % {'namespace': namespace}
        resp, body = self.http_client.get(url)
        for key, value in body['properties'].items():
            value['name'] = key
            yield (self.model(value), resp)

    @utils.add_req_id_to_object()
    def delete(self, namespace, prop_name):
        """Delete a property."""
        url = '/v2/metadefs/namespaces/%(namespace)s/properties/%(prop_name)s' % {'namespace': namespace, 'prop_name': prop_name}
        resp, body = self.http_client.delete(url)
        return ((resp, body), resp)

    @utils.add_req_id_to_object()
    def delete_all(self, namespace):
        """Delete all properties in a namespace."""
        url = '/v2/metadefs/namespaces/%(namespace)s/properties' % {'namespace': namespace}
        resp, body = self.http_client.delete(url)
        return ((resp, body), resp)