import copy
from collections import deque
from pprint import pformat
from botocore.awsrequest import AWSResponse
from botocore.exceptions import (
from botocore.validate import validate_parameters
def _should_not_stub(self, context):
    if context and context.get('is_presign_request'):
        return True