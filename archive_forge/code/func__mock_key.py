import copy
from unittest import mock
from heat.common import exception
from heat.engine.clients.os import keystone
from heat.engine.clients.os import nova
from heat.engine import resource
from heat.engine.resources.openstack.nova import keypair
from heat.engine import scheduler
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def _mock_key(self, name, pub=None, priv=None):
    mkey = mock.MagicMock()
    mkey.id = name
    mkey.name = name
    if pub:
        mkey.public_key = pub
    if priv:
        mkey.private_key = priv
    return mkey