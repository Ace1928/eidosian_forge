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
def _sse_md5(params, sse_member_prefix='SSECustomer'):
    if not _needs_s3_sse_customization(params, sse_member_prefix):
        return
    sse_key_member = sse_member_prefix + 'Key'
    sse_md5_member = sse_member_prefix + 'KeyMD5'
    key_as_bytes = params[sse_key_member]
    if isinstance(key_as_bytes, str):
        key_as_bytes = key_as_bytes.encode('utf-8')
    key_md5_str = base64.b64encode(get_md5(key_as_bytes).digest()).decode('utf-8')
    key_b64_encoded = base64.b64encode(key_as_bytes).decode('utf-8')
    params[sse_key_member] = key_b64_encoded
    params[sse_md5_member] = key_md5_str