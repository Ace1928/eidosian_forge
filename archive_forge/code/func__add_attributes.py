from botocore import xform_name
from botocore.docs.utils import get_official_service_name
from boto3.docs.base import BaseDocumenter
from boto3.docs.action import ActionDocumenter
from boto3.docs.waiter import WaiterResourceDocumenter
from boto3.docs.collection import CollectionDocumenter
from boto3.docs.subresource import SubResourceDocumenter
from boto3.docs.attr import document_attribute
from boto3.docs.attr import document_identifier
from boto3.docs.attr import document_reference
from boto3.docs.utils import get_identifier_args_for_signature
from boto3.docs.utils import get_identifier_values_for_example
from boto3.docs.utils import get_identifier_description
from boto3.docs.utils import add_resource_type_overview
def _add_attributes(self, section):
    service_model = self._resource.meta.client.meta.service_model
    attributes = {}
    if self._resource.meta.resource_model.shape:
        shape = service_model.shape_for(self._resource.meta.resource_model.shape)
        attributes = self._resource.meta.resource_model.get_attributes(shape)
    section = section.add_new_section('attributes')
    attribute_list = []
    if attributes:
        add_resource_type_overview(section=section, resource_type='Attributes', description='Attributes provide access to the properties of a resource. Attributes are lazy-loaded the first time one is accessed via the :py:meth:`load` method.', intro_link='identifiers_attributes_intro')
        self.member_map['attributes'] = attribute_list
    for attr_name in sorted(attributes):
        _, attr_shape = attributes[attr_name]
        attribute_section = section.add_new_section(attr_name)
        attribute_list.append(attr_name)
        document_attribute(section=attribute_section, service_name=self._service_name, resource_name=self._resource_name, attr_name=attr_name, event_emitter=self._resource.meta.client.meta.events, attr_model=attr_shape)