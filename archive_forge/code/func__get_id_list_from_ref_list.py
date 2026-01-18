import datetime
import freezegun
import http.client
from oslo_config import fixture as config_fixture
from oslo_serialization import jsonutils
from keystone.common import provider_api
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import filtering
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
from keystone.tests.unit import test_v3
def _get_id_list_from_ref_list(self, ref_list):
    result_list = []
    for x in ref_list:
        result_list.append(x['id'])
    return result_list