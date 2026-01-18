import platform
import socket
import sys
from oslo_log import log as logging
from oslo_utils import importutils
from os_brick import exception
from os_brick.i18n import _
from os_brick import initiator
from os_brick import utils
def _get_connector_list():
    if sys.platform != 'win32':
        return unix_connector_list
    else:
        return windows_connector_list