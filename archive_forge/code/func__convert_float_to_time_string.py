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
def _convert_float_to_time_string(cls, time_float):
    """Convert a floating point timestamp to a string.

        :param time_float: integer representing timestamp
        :returns: a time formatted strings

        """
    time_object = datetime.datetime.utcfromtimestamp(time_float)
    return ks_utils.isotime(time_object, subsecond=True)