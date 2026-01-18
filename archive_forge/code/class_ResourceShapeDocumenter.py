from botocore.docs.params import ResponseParamsDocumenter
from boto3.docs.utils import get_identifier_description
class ResourceShapeDocumenter(ResponseParamsDocumenter):
    EVENT_NAME = 'resource-shape'