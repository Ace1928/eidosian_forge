from unittest import mock
from oslo_config import cfg
from oslotest import base
from cinderclient import exceptions as cinder_exception
from glance_store.common import attachment_state_manager as attach_manager
from glance_store.common import cinder_utils
from glance_store import exceptions
def _sentinel_attach(self):
    attachment_id = self.m.attach(mock.sentinel.client, mock.sentinel.volume_id, mock.sentinel.host, mode=mock.sentinel.mode)
    return attachment_id