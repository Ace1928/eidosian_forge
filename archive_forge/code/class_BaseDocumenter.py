from botocore.compat import OrderedDict
class BaseDocumenter(object):

    def __init__(self, resource):
        self._resource = resource
        self._client = self._resource.meta.client
        self._resource_model = self._resource.meta.resource_model
        self._service_model = self._client.meta.service_model
        self._resource_name = self._resource.meta.resource_model.name
        self._service_name = self._service_model.service_name
        self._service_docs_name = self._client.__class__.__name__
        self.member_map = OrderedDict()
        self.represents_service_resource = self._service_name == self._resource_name

    @property
    def class_name(self):
        return '%s.%s' % (self._service_docs_name, self._resource_name)