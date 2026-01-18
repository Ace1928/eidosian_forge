import functools
import logging
import re
from oslo_utils import strutils
from cinderclient._i18n import _
from cinderclient import exceptions
from cinderclient import utils
def _validate_requested_version(requested_version, server_start_version, server_end_version):
    """Validates the requested version.

    Checks 'requested_version' is within the min/max range supported by the
    server. If 'requested_version' is not within range then attempts to
    downgrade to 'server_end_version'. Otherwise an UnsupportedVersion
    exception is thrown.

    :param requested_version: requestedversion represented by APIVersion obj
    :param server_start_version: APIVersion object representing server min
    :param server_end_version: APIVersion object representing server max
    """
    valid_version = requested_version
    if not requested_version.matches(server_start_version, server_end_version):
        if server_end_version <= requested_version:
            if APIVersion(MIN_VERSION) <= server_end_version and server_end_version <= APIVersion(MAX_VERSION):
                msg = _('Requested version %(requested_version)s is not supported. Downgrading requested version to %(server_end_version)s.')
                LOG.debug(msg, {'requested_version': requested_version, 'server_end_version': server_end_version})
            valid_version = server_end_version
        else:
            raise exceptions.UnsupportedVersion(_("The specified version isn't supported by server. The valid version range is '%(min)s' to '%(max)s'") % {'min': server_start_version.get_string(), 'max': server_end_version.get_string()})
    return valid_version