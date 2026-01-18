from the current environment without the need to copy, save and manage
import abc
import copy
from dataclasses import dataclass
import datetime
import io
import json
import re
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.auth import impersonated_credentials
from google.auth import metrics
from google.oauth2 import sts
from google.oauth2 import utils
@dataclass
class SupplierContext:
    """A context class that contains information about the requested third party credential that is passed
        to AWS security credential and subject token suppliers.

        Attributes:
            subject_token_type (str): The requested subject token type based on the Oauth2.0 token exchange spec.
                Expected values include::

                    “urn:ietf:params:oauth:token-type:jwt”
                    “urn:ietf:params:oauth:token-type:id-token”
                    “urn:ietf:params:oauth:token-type:saml2”
                    “urn:ietf:params:aws:token-type:aws4_request”

            audience (str): The requested audience for the subject token.
    """
    subject_token_type: str
    audience: str