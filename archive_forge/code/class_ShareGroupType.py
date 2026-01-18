import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
class ShareGroupType(object):
    id = 'fake group type id'
    name = 'fake group type name'
    share_types = [ShareType().id]
    is_public = False