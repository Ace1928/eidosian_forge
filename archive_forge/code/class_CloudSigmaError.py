import re
import copy
import time
import base64
import hashlib
from libcloud.utils.py3 import b, httplib
from libcloud.utils.misc import dict2str, str2list, str2dicts, get_secure_random_string
from libcloud.common.base import Response, JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError, InvalidCredsError
from libcloud.compute.base import Node, KeyPair, NodeSize, NodeImage, NodeDriver, is_private_subnet
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.cloudsigma import (
class CloudSigmaError(ProviderError):
    """
    Represents CloudSigma API error.
    """

    def __init__(self, http_code, error_type, error_msg, error_point, driver):
        """
        :param http_code: HTTP status code.
        :type http_code: ``int``

        :param error_type: Type of error (validation / notexist / backend /
                           permissions  database / concurrency / billing /
                           payment)
        :type error_type: ``str``

        :param error_msg: A description of the error that occurred.
        :type error_msg: ``str``

        :param error_point: Point at which the error occurred. Can be None.
        :type error_point: ``str`` or ``None``
        """
        super().__init__(http_code=http_code, value=error_msg, driver=driver)
        self.error_type = error_type
        self.error_msg = error_msg
        self.error_point = error_point