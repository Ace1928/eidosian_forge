import copy
import json
import email.utils
from typing import Dict, Optional
from libcloud.utils.py3 import httplib, urlquote
from libcloud.common.base import ConnectionUserAndKey
from libcloud.common.types import ProviderError
from libcloud.common.google import GoogleAuthType, GoogleResponse, GoogleOAuth2Credential
from libcloud.storage.drivers.s3 import (
class GCSResponse(GoogleResponse):
    pass