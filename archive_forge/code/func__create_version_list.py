import copy
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
import testtools
import barbicanclient.barbican
def _create_version_list(versions):
    return jsonutils.dumps({'versions': {'values': versions}})