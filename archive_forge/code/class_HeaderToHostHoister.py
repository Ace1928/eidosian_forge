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
class HeaderToHostHoister:
    """Takes a header and moves it to the front of the hoststring."""
    _VALID_HOSTNAME = re.compile('(?!-)[a-z\\d-]{1,63}(?<!-)$', re.IGNORECASE)

    def __init__(self, header_name):
        self._header_name = header_name

    def hoist(self, params, **kwargs):
        """Hoist a header to the hostname.

        Hoist a header to the beginning of the hostname with a suffix "." after
        it. The original header should be removed from the header map. This
        method is intended to be used as a target for the before-call event.
        """
        if self._header_name not in params['headers']:
            return
        header_value = params['headers'][self._header_name]
        self._ensure_header_is_valid_host(header_value)
        original_url = params['url']
        new_url = self._prepend_to_host(original_url, header_value)
        params['url'] = new_url

    def _ensure_header_is_valid_host(self, header):
        match = self._VALID_HOSTNAME.match(header)
        if not match:
            raise ParamValidationError(report='Hostnames must contain only - and alphanumeric characters, and between 1 and 63 characters long.')

    def _prepend_to_host(self, url, prefix):
        url_components = urlsplit(url)
        parts = url_components.netloc.split('.')
        parts = [prefix] + parts
        new_netloc = '.'.join(parts)
        new_components = (url_components.scheme, new_netloc, url_components.path, url_components.query, '')
        new_url = urlunsplit(new_components)
        return new_url