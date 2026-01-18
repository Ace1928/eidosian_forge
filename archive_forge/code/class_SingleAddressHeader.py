from types import MappingProxyType
from email import utils
from email import errors
from email import _header_value_parser as parser
class SingleAddressHeader(AddressHeader):

    @property
    def address(self):
        if len(self.addresses) != 1:
            raise ValueError('value of single address header {} is not a single address'.format(self.name))
        return self.addresses[0]