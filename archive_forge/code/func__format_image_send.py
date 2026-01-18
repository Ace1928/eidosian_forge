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
def _format_image_send(self, bytes_sent):
    return {'bytes_sent': bytes_sent, 'image_id': self.repo.image_id, 'owner_id': self.repo.owner, 'receiver_tenant_id': self.context.project_id, 'receiver_user_id': self.context.user_id}