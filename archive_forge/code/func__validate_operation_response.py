import copy
from collections import deque
from pprint import pformat
from botocore.awsrequest import AWSResponse
from botocore.exceptions import (
from botocore.validate import validate_parameters
def _validate_operation_response(self, operation_name, service_response):
    service_model = self.client.meta.service_model
    operation_model = service_model.operation_model(operation_name)
    output_shape = operation_model.output_shape
    response = service_response
    if 'ResponseMetadata' in response:
        response = copy.copy(service_response)
        del response['ResponseMetadata']
    self._validate_response(output_shape, response)