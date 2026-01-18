import copy
from collections import deque
from pprint import pformat
from botocore.awsrequest import AWSResponse
from botocore.exceptions import (
from botocore.validate import validate_parameters
def _get_response_handler(self, model, params, context, **kwargs):
    self._assert_expected_call_order(model, params)
    return self._queue.popleft()['response']