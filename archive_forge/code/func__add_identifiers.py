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
def _add_identifiers(self, section):
    identifiers = self._resource.meta.resource_model.identifiers
    section = section.add_new_section('identifiers')
    member_list = []
    if identifiers:
        self.member_map['identifiers'] = member_list
        add_resource_type_overview(section=section, resource_type='Identifiers', description='Identifiers are properties of a resource that are set upon instantation of the resource.', intro_link='identifiers_attributes_intro')
    for identifier in identifiers:
        identifier_section = section.add_new_section(identifier.name)
        member_list.append(identifier.name)
        document_identifier(section=identifier_section, resource_name=self._resource_name, identifier_model=identifier)