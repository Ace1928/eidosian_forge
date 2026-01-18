import datetime
import os
import jwt
from oslo_utils import timeutils
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.token.providers import base
def _convert_time_string_to_int(self, time_str):
    time_object = timeutils.parse_isotime(time_str)
    normalized = timeutils.normalize_time(time_object)
    epoch = datetime.datetime.utcfromtimestamp(0)
    return int((normalized - epoch).total_seconds())