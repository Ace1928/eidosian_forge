import copy
import json
import email.utils
from typing import Dict, Optional
from libcloud.utils.py3 import httplib, urlquote
from libcloud.common.base import ConnectionUserAndKey
from libcloud.common.types import ProviderError
from libcloud.common.google import GoogleAuthType, GoogleResponse, GoogleOAuth2Credential
from libcloud.storage.drivers.s3 import (
class GoogleStorageJSONConnection(GoogleStorageConnection):
    """
    Represents a single connection to the Google storage JSON API endpoint.

    This can either authenticate via the Google OAuth2 methods or via
    the S3 HMAC interoperability method.
    """
    host = 'www.googleapis.com'
    responseCls = GCSResponse
    rawResponseCls = None

    def add_default_headers(self, headers):
        headers = super().add_default_headers(headers)
        headers['Content-Type'] = 'application/json'
        return headers