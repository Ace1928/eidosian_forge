import contextlib
import hashlib
import io
import math
import os
from unittest import mock
import socket
import sys
import tempfile
import time
import uuid
from keystoneauth1 import exceptions as keystone_exc
from os_brick.initiator import connector
from oslo_concurrency import processutils
from oslo_utils.secretutils import md5
from oslo_utils import units
from glance_store._drivers.cinder import scaleio
from glance_store.common import attachment_state_manager
from glance_store.common import cinder_utils
from glance_store import exceptions
from glance_store import location
from glance_store._drivers.cinder import store as cinder # noqa
from glance_store._drivers.cinder import nfs # noqa
def _test_get_cinderclient_cinder_endpoint_template(self, group='glance_store'):
    fake_endpoint = 'http://cinder.openstack.example.com/v2/fake_project'
    self.config(cinder_endpoint_template=fake_endpoint, group=group)
    with mock.patch.object(cinder.ksa_token_endpoint, 'Token') as fake_token:
        self.store.get_cinderclient(self.context)
        fake_token.assert_called_once_with(endpoint=fake_endpoint, token=self.context.auth_token)