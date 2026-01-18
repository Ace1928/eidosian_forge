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
def _decode_list_object(top_level_keys, nested_keys, parsed, context):
    if parsed.get('EncodingType') == 'url' and context.get('encoding_type_auto_set'):
        for key in top_level_keys:
            if key in parsed:
                parsed[key] = unquote_str(parsed[key])
        for top_key, child_key in nested_keys:
            if top_key in parsed:
                for member in parsed[top_key]:
                    member[child_key] = unquote_str(member[child_key])