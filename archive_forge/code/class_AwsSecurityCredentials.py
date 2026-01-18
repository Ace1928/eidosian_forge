import abc
from dataclasses import dataclass
import hashlib
import hmac
import http.client as http_client
import json
import os
import posixpath
import re
from typing import Optional
import urllib
from urllib.parse import urljoin
from google.auth import _helpers
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import external_account
@dataclass
class AwsSecurityCredentials:
    """A class that models AWS security credentials with an optional session token.

        Attributes:
            access_key_id (str): The AWS security credentials access key id.
            secret_access_key (str): The AWS security credentials secret access key.
            session_token (Optional[str]): The optional AWS security credentials session token. This should be set when using temporary credentials.
    """
    access_key_id: str
    secret_access_key: str
    session_token: Optional[str] = None