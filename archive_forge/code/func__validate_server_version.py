import functools
import logging
import re
from oslo_utils import strutils
from cinderclient._i18n import _
from cinderclient import exceptions
from cinderclient import utils
def _validate_server_version(server_start_version, server_end_version):
    """Validates the server version.

    Checks that the 'server_end_version' is greater than the minimum version
    supported by the client. Then checks that the 'server_start_version' is
    less than the maximum version supported by the client.

    :param server_start_version:
    :param server_end_version:
    :return:
    """
    if APIVersion(MIN_VERSION) > server_end_version:
        raise exceptions.UnsupportedVersion(_("Server's version is too old. The client's valid version range is '%(client_min)s' to '%(client_max)s'. The server valid version range is '%(server_min)s' to '%(server_max)s'.") % {'client_min': MIN_VERSION, 'client_max': MAX_VERSION, 'server_min': server_start_version.get_string(), 'server_max': server_end_version.get_string()})
    elif APIVersion(MAX_VERSION) < server_start_version:
        raise exceptions.UnsupportedVersion(_("Server's version is too new. The client's valid version range is '%(client_min)s' to '%(client_max)s'. The server valid version range is '%(server_min)s' to '%(server_max)s'.") % {'client_min': MIN_VERSION, 'client_max': MAX_VERSION, 'server_min': server_start_version.get_string(), 'server_max': server_end_version.get_string()})