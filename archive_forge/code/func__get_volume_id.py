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
def _get_volume_id(self):
    volname_encoded = urllib.parse.quote(self.volume_name, '')
    volname_double_encoded = urllib.parse.quote(volname_encoded, '')
    LOG.debug(_('Volume name after double encoding is %(volume_name)s.'), {'volume_name': volname_double_encoded})
    request = 'https://%(server_ip)s:%(server_port)s/api/types/Volume/instances/getByName::%(encoded_volume_name)s' % {'server_ip': self.server_ip, 'server_port': self.server_port, 'encoded_volume_name': volname_double_encoded}
    LOG.info('ScaleIO get volume id by name request: %(request)s', {'request': request})
    r = requests.get(request, auth=(self.server_username, self.server_token), verify=self._verify_cert())
    r = self._check_response(r, request)
    volume_id = r.json()
    if not volume_id:
        msg = _("Volume with name %(volume_name)s wasn't found.") % {'volume_name': self.volume_name}
        LOG.error(msg)
        raise exception.BrickException(message=msg)
    if r.status_code != self.OK_STATUS_CODE and 'errorCode' in volume_id:
        msg = _('Error getting volume id from name %(volume_name)s: %(err)s') % {'volume_name': self.volume_name, 'err': volume_id['message']}
        LOG.error(msg)
        raise exception.BrickException(message=msg)
    LOG.info('ScaleIO volume id is %(volume_id)s.', {'volume_id': volume_id})
    return volume_id