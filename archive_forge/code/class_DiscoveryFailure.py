import os_service_types
from keystoneauth1.exceptions import base
class DiscoveryFailure(base.ClientException):
    message = 'Discovery of client versions failed.'