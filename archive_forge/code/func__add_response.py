import copy
from collections import deque
from pprint import pformat
from botocore.awsrequest import AWSResponse
from botocore.exceptions import (
from botocore.validate import validate_parameters
def _add_response(self, method, service_response, expected_params):
    if not hasattr(self.client, method):
        raise ValueError('Client %s does not have method: %s' % (self.client.meta.service_model.service_name, method))
    http_response = AWSResponse(None, 200, {}, None)
    operation_name = self.client.meta.method_to_api_mapping.get(method)
    self._validate_operation_response(operation_name, service_response)
    response = {'operation_name': operation_name, 'response': (http_response, service_response), 'expected_params': expected_params}
    self._queue.append(response)