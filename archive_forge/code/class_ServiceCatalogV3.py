import abc
import copy
from keystoneauth1 import discover
from keystoneauth1 import exceptions
class ServiceCatalogV3(ServiceCatalog):
    """An object for encapsulating the v3 service catalog.

    The object is created using raw v3 auth token from Keystone.
    """

    @classmethod
    def from_token(cls, token):
        if 'token' not in token:
            raise ValueError('Invalid token format for fetching catalog')
        return cls(token['token'].get('catalog', {}))

    @staticmethod
    def normalize_interface(interface):
        if interface:
            interface = interface.rstrip('URL')
        return interface

    def is_interface_match(self, endpoint, interface):
        try:
            return interface == endpoint['interface']
        except KeyError:
            return False