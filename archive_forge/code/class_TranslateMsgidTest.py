import logging
from unittest import mock
import warnings
from oslotest import base as test_base
import testtools
from oslo_i18n import _message
from oslo_i18n.tests import fakes
from oslo_i18n.tests import utils
class TranslateMsgidTest(test_base.BaseTestCase):

    @mock.patch('gettext.translation')
    def test_contextual(self, translation):
        lang = mock.Mock()
        translation.return_value = lang
        trans = mock.Mock()
        trans.return_value = 'translated'
        lang.gettext = trans
        lang.ugettext = trans
        result = _message.Message._translate_msgid(('context', 'message'), domain='domain', has_contextual_form=True, has_plural_form=False)
        self.assertEqual('translated', result)
        trans.assert_called_with('context' + _message.CONTEXT_SEPARATOR + 'message')

    @mock.patch('gettext.translation')
    def test_contextual_untranslatable(self, translation):
        msg_with_context = 'context' + _message.CONTEXT_SEPARATOR + 'message'
        lang = mock.Mock()
        translation.return_value = lang
        trans = mock.Mock()
        trans.return_value = msg_with_context
        lang.gettext = trans
        lang.ugettext = trans
        result = _message.Message._translate_msgid(('context', 'message'), domain='domain', has_contextual_form=True, has_plural_form=False)
        self.assertEqual('message', result)
        trans.assert_called_with(msg_with_context)

    @mock.patch('gettext.translation')
    def test_plural(self, translation):
        lang = mock.Mock()
        translation.return_value = lang
        trans = mock.Mock()
        trans.return_value = 'translated'
        lang.ngettext = trans
        lang.ungettext = trans
        result = _message.Message._translate_msgid(('single', 'plural', -1), domain='domain', has_contextual_form=False, has_plural_form=True)
        self.assertEqual('translated', result)
        trans.assert_called_with('single', 'plural', -1)

    @mock.patch('gettext.translation')
    def test_contextual_and_plural(self, translation):
        self.assertRaises(ValueError, _message.Message._translate_msgid, 'nothing', domain='domain', has_contextual_form=True, has_plural_form=True)