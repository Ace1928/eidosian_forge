import abc
import requests
import requests.auth
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.identity import v3
class ConsumerMismatch(SamlException):
    """The SP and IDP consumers do not match."""