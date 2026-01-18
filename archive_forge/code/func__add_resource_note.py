import os
from botocore import xform_name
from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.utils import get_official_service_name
from boto3.docs.action import ActionDocumenter
from boto3.docs.attr import (
from boto3.docs.base import BaseDocumenter
from boto3.docs.collection import CollectionDocumenter
from boto3.docs.subresource import SubResourceDocumenter
from boto3.docs.utils import (
from boto3.docs.waiter import WaiterResourceDocumenter
def _add_resource_note(self, section):
    section = section.add_new_section('feature-freeze')
    section.style.start_note()
    section.write('Before using anything on this page, please refer to the resources :doc:`user guide <../../../../guide/resources>` for the most recent guidance on using resources.')
    section.style.end_note()