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
def _quote_source_header_from_dict(source_dict):
    try:
        bucket = source_dict['Bucket']
        key = source_dict['Key']
        version_id = source_dict.get('VersionId')
        if VALID_S3_ARN.search(bucket):
            final = f'{bucket}/object/{key}'
        else:
            final = f'{bucket}/{key}'
    except KeyError as e:
        raise ParamValidationError(report=f'Missing required parameter: {str(e)}')
    final = percent_encode(final, safe=SAFE_CHARS + '/')
    if version_id is not None:
        final += '?versionId=%s' % version_id
    return final