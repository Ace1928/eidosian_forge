from unittest import mock
from oslo_config import cfg
from glance import context
from glance.tests.unit import utils as unit_utils
from glance.tests import utils
def _fake_image(owner, is_public):
    return {'id': None, 'owner': owner, 'visibility': 'public' if is_public else 'shared'}