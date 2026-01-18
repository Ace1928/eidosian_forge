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
def _add_references(self, section):
    section = section.add_new_section('references')
    references = self._resource.meta.resource_model.references
    reference_list = []
    if references:
        add_resource_type_overview(section=section, resource_type='References', description='References are related resource instances that have a belongs-to relationship.', intro_link='references_intro')
        self.member_map['references'] = reference_list
    for reference in references:
        reference_section = section.add_new_section(reference.name)
        reference_list.append(reference.name)
        document_reference(section=reference_section, reference_model=reference)