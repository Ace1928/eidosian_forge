import copy
from collections import deque
from pprint import pformat
from botocore.awsrequest import AWSResponse
from botocore.exceptions import (
from botocore.validate import validate_parameters
def _assert_expected_call_order(self, model, params):
    if not self._queue:
        raise UnStubbedResponseError(operation_name=model.name, reason='Unexpected API Call: A call was made but no additional calls expected. Either the API Call was not stubbed or it was called multiple times.')
    name = self._queue[0]['operation_name']
    if name != model.name:
        raise StubResponseError(operation_name=model.name, reason=f'Operation mismatch: found response for {name}.')