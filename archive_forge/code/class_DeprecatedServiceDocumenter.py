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
class DeprecatedServiceDocumenter:

    def __init__(self, replacement_service_name):
        self._replacement_service_name = replacement_service_name

    def inject_deprecation_notice(self, section, event_name, **kwargs):
        section.style.start_important()
        section.write('This service client is deprecated. Please use ')
        section.style.ref(self._replacement_service_name, self._replacement_service_name)
        section.write(' instead.')
        section.style.end_important()