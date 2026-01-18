import base64
import copy
import logging
import os
import re
import uuid
import warnings
from io import BytesIO
import botocore
import botocore.auth
from botocore import utils
from botocore.compat import (
from botocore.docs.utils import (
from botocore.endpoint_provider import VALID_HOST_LABEL_RE
from botocore.exceptions import (
from botocore.regions import EndpointResolverBuiltins
from botocore.signers import (
from botocore.utils import (
from botocore import retryhandler  # noqa
from botocore import translate  # noqa
from botocore.compat import MD5_AVAILABLE  # noqa
from botocore.exceptions import MissingServiceIdError  # noqa
from botocore.utils import hyphenize_service_id  # noqa
from botocore.utils import is_global_accesspoint  # noqa
from botocore.utils import SERVICE_NAME_ALIASES  # noqa
def add_glacier_checksums(params, **kwargs):
    """Add glacier checksums to the http request.

    This will add two headers to the http request:

        * x-amz-content-sha256
        * x-amz-sha256-tree-hash

    These values will only be added if they are not present
    in the HTTP request.

    """
    request_dict = params
    headers = request_dict['headers']
    body = request_dict['body']
    if isinstance(body, bytes):
        body = BytesIO(body)
    starting_position = body.tell()
    if 'x-amz-content-sha256' not in headers:
        headers['x-amz-content-sha256'] = utils.calculate_sha256(body, as_hex=True)
    body.seek(starting_position)
    if 'x-amz-sha256-tree-hash' not in headers:
        headers['x-amz-sha256-tree-hash'] = utils.calculate_tree_hash(body)
    body.seek(starting_position)