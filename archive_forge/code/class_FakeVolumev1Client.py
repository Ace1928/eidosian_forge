import copy
import random
from unittest import mock
import uuid
from openstack.image.v1 import _proxy as image_v1_proxy
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
from openstackclient.tests.unit import utils
class FakeVolumev1Client:

    def __init__(self, **kwargs):
        self.volumes = mock.Mock()
        self.volumes.resource_class = fakes.FakeResource(None, {})
        self.services = mock.Mock()
        self.services.resource_class = fakes.FakeResource(None, {})
        self.extensions = mock.Mock()
        self.extensions.resource_class = fakes.FakeResource(None, {})
        self.qos_specs = mock.Mock()
        self.qos_specs.resource_class = fakes.FakeResource(None, {})
        self.volume_types = mock.Mock()
        self.volume_types.resource_class = fakes.FakeResource(None, {})
        self.volume_encryption_types = mock.Mock()
        self.volume_encryption_types.resource_class = fakes.FakeResource(None, {})
        self.transfers = mock.Mock()
        self.transfers.resource_class = fakes.FakeResource(None, {})
        self.volume_snapshots = mock.Mock()
        self.volume_snapshots.resource_class = fakes.FakeResource(None, {})
        self.backups = mock.Mock()
        self.backups.resource_class = fakes.FakeResource(None, {})
        self.restores = mock.Mock()
        self.restores.resource_class = fakes.FakeResource(None, {})
        self.auth_token = kwargs['token']
        self.management_url = kwargs['endpoint']