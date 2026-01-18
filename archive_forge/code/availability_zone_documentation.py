from oslo_serialization import jsonutils
from neutron_lib._i18n import _
from neutron_lib.api import validators
from neutron_lib.db import constants as db_const
from neutron_lib import exceptions
Convert an AZ list in string format into a python list.

    :param az_string: The AZ list in string format.
    :returns: The python list of AZs build from az_string.
    