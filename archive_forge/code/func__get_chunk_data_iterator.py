import abc
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
import oslo_messaging
from oslo_utils import encodeutils
from oslo_utils import excutils
import webob
from glance.common import exception
from glance.common import timeutils
from glance.domain import proxy as domain_proxy
from glance.i18n import _, _LE
def _get_chunk_data_iterator(self, data, chunk_size=None):
    sent = 0
    for chunk in data:
        yield chunk
        sent += len(chunk)
    if sent != (chunk_size or self.repo.size):
        notify = self.notifier.error
    else:
        notify = self.notifier.info
    try:
        _send_notification(notify, 'image.send', self._format_image_send(sent))
    except Exception as err:
        msg = _LE('An error occurred during image.send notification: %(err)s') % {'err': err}
        LOG.error(msg)