from oslo_serialization import jsonutils
from neutron_lib._i18n import _
from neutron_lib.api import validators
from neutron_lib.db import constants as db_const
from neutron_lib import exceptions
def _validate_availability_zone_hints(data, valid_value=None):
    msg = validators.validate_list_of_unique_strings(data)
    if msg:
        return msg
    az_string = convert_az_list_to_string(data)
    if len(az_string) > db_const.AZ_HINTS_DB_LEN:
        msg = _('Too many availability_zone_hints specified')
        raise exceptions.InvalidInput(error_message=msg)