import os
import boto3
from botocore.exceptions import DataNotFoundError
from botocore.docs.service import ServiceDocumenter as BaseServiceDocumenter
from botocore.docs.bcdoc.restdoc import DocumentStructure
from boto3.utils import ServiceContext
from boto3.docs.client import Boto3ClientDocumenter
from boto3.docs.resource import ResourceDocumenter
from boto3.docs.resource import ServiceResourceDocumenter
def _get_example_file(self):
    return os.path.realpath(os.path.join(self.EXAMPLE_PATH, self._service_name + '.rst'))