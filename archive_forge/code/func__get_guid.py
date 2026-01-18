import json
import os
import urllib
from oslo_log import log as logging
import requests
from os_brick import exception
from os_brick.i18n import _
from os_brick import initiator
from os_brick.initiator.connectors import base
from os_brick.privileged import scaleio as priv_scaleio
from os_brick import utils
def _get_guid(self):
    try:
        guid = priv_scaleio.get_guid(self.GET_GUID_OP_CODE)
        LOG.info('Current sdc guid: %s', guid)
        return guid
    except (IOError, OSError, ValueError) as e:
        msg = _('Error querying sdc guid: %s') % e
        LOG.error(msg)
        raise exception.BrickException(message=msg)