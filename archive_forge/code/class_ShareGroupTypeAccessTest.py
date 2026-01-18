from unittest import mock
import ddt
import manilaclient
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes as fake
from manilaclient.v2 import share_group_type_access as type_access
@ddt.ddt
class ShareGroupTypeAccessTest(utils.TestCase):

    def setUp(self):
        super(ShareGroupTypeAccessTest, self).setUp()
        self.manager = type_access.ShareGroupTypeAccessManager(fake.FakeClient())
        fake_group_type_access_info = {'share_group_type_id': fake.ShareGroupTypeAccess.id}
        self.share_group_type_access = type_access.ShareGroupTypeAccess(self.manager, fake_group_type_access_info, loaded=True)

    def test_repr(self):
        result = str(self.share_group_type_access)
        self.assertEqual('<Share Group Type Access: %s>' % fake.ShareGroupTypeAccess.id, result)