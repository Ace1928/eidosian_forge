import os
import re
import binascii
from typing import IO, List, Union, Optional, cast
from libcloud.utils.py3 import basestring
from libcloud.compute.ssh import BaseSSHClient
from libcloud.compute.base import Node
def _get_string_value(self, argument_name, argument_value):
    if not isinstance(argument_value, basestring) and (not hasattr(argument_value, 'read')):
        raise TypeError('%s argument must be a string or a file-like object' % argument_name)
    if hasattr(argument_value, 'read'):
        argument_value = argument_value.read()
    return argument_value