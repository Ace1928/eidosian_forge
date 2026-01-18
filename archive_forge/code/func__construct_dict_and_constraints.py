import string
from unittest import mock
import netaddr
from neutron_lib._i18n import _
from neutron_lib.api import converters
from neutron_lib.api.definitions import extra_dhcp_opt
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib import fixture
from neutron_lib.plugins import directory
from neutron_lib.tests import _base as base
def _construct_dict_and_constraints(self):
    """Constructs a test dictionary and a definition of constraints.

        :return: A (dictionary, constraint) tuple
        """
    constraints = {'key1': {'type:values': ['val1', 'val2'], 'required': True}, 'key2': {'type:string': None, 'required': False}, 'key3': {'type:dict': {'k4': {'type:string': None, 'required': True}}, 'required': True}}
    dictionary = {'key1': 'val1', 'key2': 'a string value', 'key3': {'k4': 'a string value'}}
    return (dictionary, constraints)