import os
import copy
import hmac
import time
import base64
from hashlib import sha256
from libcloud.http import LibcloudConnection
from libcloud.utils.py3 import ET, b, httplib, urlparse, urlencode, basestring
from libcloud.utils.xml import fixxpath
from libcloud.common.base import (
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.common.azure_arm import AzureAuthJsonResponse, publicEnvironments
class AzureRawResponse(RawResponse):
    pass