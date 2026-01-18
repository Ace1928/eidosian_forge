import abc
import requests
import requests.auth
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.identity import v3
class SamlException(Exception):
    """Base SAML plugin exception."""