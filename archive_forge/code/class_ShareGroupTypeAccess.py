import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
class ShareGroupTypeAccess(object):
    id = 'fake group type access id'
    name = 'fake group type access name'