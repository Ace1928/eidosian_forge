from botocore.docs.docstring import LazyLoadedDocstring
from boto3.docs.action import document_action
from boto3.docs.action import document_load_reload_action
from boto3.docs.subresource import document_sub_resource
from boto3.docs.attr import document_attribute
from boto3.docs.attr import document_identifier
from boto3.docs.attr import document_reference
from boto3.docs.collection import document_collection_object
from boto3.docs.collection import document_collection_method
from boto3.docs.collection import document_batch_action
from boto3.docs.waiter import document_resource_waiter
class AttributeDocstring(LazyLoadedDocstring):

    def _write_docstring(self, *args, **kwargs):
        document_attribute(*args, **kwargs)