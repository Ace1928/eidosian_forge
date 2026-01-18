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
def _get_cross_region_presigned_url(request_signer, request_dict, model, source_region, destination_region):
    request_dict_copy = copy.deepcopy(request_dict)
    request_dict_copy['body']['DestinationRegion'] = destination_region
    request_dict_copy['url'] = request_dict['url'].replace(destination_region, source_region)
    request_dict_copy['method'] = 'GET'
    request_dict_copy['headers'] = {}
    return request_signer.generate_presigned_url(request_dict_copy, region_name=source_region, operation_name=model.name)