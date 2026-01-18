from oslotest import base as test_base
import oslo_i18n
from oslo_i18n import _gettextutils
from oslo_i18n._i18n import _
from oslo_i18n import _lazy
from oslo_i18n import _message
from oslo_i18n import _translate
from oslo_i18n import fixture
class TranslationFixtureTest(test_base.BaseTestCase):

    def setUp(self):
        super(TranslationFixtureTest, self).setUp()
        self.trans_fixture = self.useFixture(fixture.Translation())

    def test_lazy(self):
        msg = self.trans_fixture.lazy('this is a lazy message')
        self.assertIsInstance(msg, _message.Message)
        self.assertEqual('this is a lazy message', msg.msgid)

    def test_immediate(self):
        msg = self.trans_fixture.immediate('this is a lazy message')
        self.assertNotIsInstance(msg, _message.Message)
        self.assertIsInstance(msg, str)
        self.assertEqual('this is a lazy message', msg)