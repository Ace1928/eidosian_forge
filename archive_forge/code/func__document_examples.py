import os
import boto3
from botocore.exceptions import DataNotFoundError
from botocore.docs.service import ServiceDocumenter as BaseServiceDocumenter
from botocore.docs.bcdoc.restdoc import DocumentStructure
from boto3.utils import ServiceContext
from boto3.docs.client import Boto3ClientDocumenter
from boto3.docs.resource import ResourceDocumenter
from boto3.docs.resource import ServiceResourceDocumenter
def _document_examples(self, section):
    examples_file = self._get_example_file()
    if os.path.isfile(examples_file):
        section.style.h2('Examples')
        section.style.new_line()
        section.write('.. contents::\n    :local:\n    :depth: 1')
        section.style.new_line()
        section.style.new_line()
        with open(examples_file, 'r') as f:
            section.write(f.read())