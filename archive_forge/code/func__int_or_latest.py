import copy
import re
import urllib
import os_service_types
from keystoneauth1 import _utils as utils
from keystoneauth1 import exceptions
def _int_or_latest(val):
    """Convert val to an int or the special value LATEST.

    :param val: An int()-able, or the string 'latest', or the special value
                LATEST.
    :return: An int, or the special value LATEST
    """
    return LATEST if val == 'latest' or val == LATEST else int(val)