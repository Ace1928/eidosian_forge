import base64
import datetime
import struct
import uuid
from cryptography import fernet
import msgpack
from oslo_log import log
from oslo_utils import timeutils
from keystone.auth import plugins as auth_plugins
from keystone.common import fernet_utils as utils
from keystone.common import utils as ks_utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
@classmethod
def _convert_time_string_to_float(cls, time_string):
    """Convert a time formatted string to a float.

        :param time_string: time formatted string
        :returns: a timestamp as a float

        """
    time_object = timeutils.parse_isotime(time_string)
    return (timeutils.normalize_time(time_object) - datetime.datetime.utcfromtimestamp(0)).total_seconds()