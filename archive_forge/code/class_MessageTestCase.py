import logging
from unittest import mock
import warnings
from oslotest import base as test_base
import testtools
from oslo_i18n import _message
from oslo_i18n.tests import fakes
from oslo_i18n.tests import utils
class MessageTestCase(test_base.BaseTestCase):
    """Unit tests for locale Message class."""

    def test_message_id_and_message_text(self):
        message = _message.Message('1')
        self.assertEqual('1', message.msgid)
        self.assertEqual('1', message)
        message = _message.Message('1', msgtext='A')
        self.assertEqual('1', message.msgid)
        self.assertEqual('A', message)

    def test_message_is_unicode(self):
        message = _message.Message('some %s') % 'message'
        self.assertIsInstance(message, str)

    @mock.patch('locale.getlocale')
    @mock.patch('gettext.translation')
    def test_create_message_non_english_default_locale(self, mock_translation, mock_locale):
        msgid = 'A message in English'
        es_translation = 'A message in Spanish'
        es_translations = {msgid: es_translation}
        translations_map = {'es': es_translations}
        translator = fakes.FakeTranslations.translator(translations_map)
        mock_translation.side_effect = translator
        mock_locale.return_value = ('es',)
        message = _message.Message(msgid)
        self.assertEqual(es_translation, message)
        self.assertEqual(es_translation, message.translation())

    def test_translation_returns_unicode(self):
        message = _message.Message('some %s') % 'message'
        self.assertIsInstance(message.translation(), str)

    def test_mod_with_named_parameters(self):
        msgid = '%(description)s\nCommand: %(cmd)s\nExit code: %(exit_code)s\nStdout: %(stdout)r\nStderr: %(stderr)r %%(something)s'
        params = {'description': 'test1', 'cmd': 'test2', 'exit_code': 'test3', 'stdout': 'test4', 'stderr': 'test5', 'something': 'trimmed'}
        result = _message.Message(msgid) % params
        expected = msgid % params
        self.assertEqual(expected, result)
        self.assertEqual(expected, result.translation())

    def test_multiple_mod_with_named_parameter(self):
        msgid = '%(description)s\nCommand: %(cmd)s\nExit code: %(exit_code)s\nStdout: %(stdout)r\nStderr: %(stderr)r'
        params = {'description': 'test1', 'cmd': 'test2', 'exit_code': 'test3', 'stdout': 'test4', 'stderr': 'test5'}
        first = _message.Message(msgid) % params
        expected = first % {}
        self.assertEqual(first.msgid, expected.msgid)
        self.assertEqual(first.params, expected.params)
        self.assertIsNot(expected, first)
        self.assertEqual(expected.translation(), first.translation())

    def test_mod_with_named_parameters_no_space(self):
        msgid = 'Request: %(method)s http://%(server)s:%(port)s%(url)s with headers %(headers)s'
        params = {'method': 'POST', 'server': 'test1', 'port': 1234, 'url': 'test2', 'headers': {'h1': 'val1'}}
        result = _message.Message(msgid) % params
        expected = msgid % params
        self.assertEqual(expected, result)
        self.assertEqual(expected, result.translation())

    def test_mod_with_dict_parameter(self):
        msgid = 'Test that we can inject a dictionary %s'
        params = {'description': 'test1'}
        result = _message.Message(msgid) % params
        expected = msgid % params
        self.assertEqual(expected, result)
        self.assertEqual(expected, result.translation())

    def test_mod_with_wrong_field_type_in_trans(self):
        msgid = 'Correct type %(arg1)s'
        params = {'arg1': 'test1'}
        with mock.patch('gettext.translation') as trans:
            trans.return_value.ugettext.return_value = msgid
            result = _message.Message(msgid) % params
            wrong_type = 'Wrong type %(arg1)d'
            trans.return_value.gettext.return_value = wrong_type
            trans_result = result.translation()
            expected = msgid % params
            self.assertEqual(expected, trans_result)

    def test_mod_with_wrong_field_type(self):
        msgid = 'Test that we handle unused args %(arg1)d'
        params = {'arg1': 'test1'}
        with testtools.ExpectedException(TypeError):
            _message.Message(msgid) % params

    def test_mod_with_missing_arg(self):
        msgid = 'Test that we handle missing args %(arg1)s %(arg2)s'
        params = {'arg1': 'test1'}
        with testtools.ExpectedException(KeyError, '.*arg2.*'):
            _message.Message(msgid) % params

    def test_mod_with_integer_parameters(self):
        msgid = 'Some string with params: %d'
        params = [0, 1, 10, 24124]
        messages = []
        results = []
        for param in params:
            messages.append(msgid % param)
            results.append(_message.Message(msgid) % param)
        for message, result in zip(messages, results):
            self.assertIsInstance(result, _message.Message)
            self.assertEqual(message, result.translation())
            result_str = '%s' % result.translation()
            self.assertEqual(result_str, message)
            self.assertEqual(message, result)

    def test_mod_copies_parameters(self):
        msgid = 'Found object: %(current_value)s'
        changing_dict = {'current_value': 1}
        result = _message.Message(msgid) % changing_dict
        changing_dict['current_value'] = 2
        self.assertEqual('Found object: 1', result.translation())

    def test_mod_deep_copies_parameters(self):
        msgid = 'Found list: %(current_list)s'
        changing_list = list([1, 2, 3])
        params = {'current_list': changing_list}
        result = _message.Message(msgid) % params
        changing_list.append(4)
        self.assertEqual('Found list: [1, 2, 3]', result.translation())

    def test_mod_deep_copies_param_nodeep_param(self):
        msgid = 'Value: %s'
        params = utils.NoDeepCopyObject(5)
        result = _message.Message(msgid) % params
        self.assertEqual('Value: 5', result.translation())

    def test_mod_deep_copies_param_nodeep_dict(self):
        msgid = 'Values: %(val1)s %(val2)s'
        params = {'val1': 1, 'val2': utils.NoDeepCopyObject(2)}
        result = _message.Message(msgid) % params
        self.assertEqual('Values: 1 2', result.translation())
        params = {'val1': 3, 'val2': utils.NoDeepCopyObject(4)}
        result = _message.Message(msgid) % params
        self.assertEqual('Values: 3 4', result.translation())

    def test_mod_returns_a_copy(self):
        msgid = 'Some msgid string: %(test1)s %(test2)s'
        message = _message.Message(msgid)
        m1 = message % {'test1': 'foo', 'test2': 'bar'}
        m2 = message % {'test1': 'foo2', 'test2': 'bar2'}
        self.assertIsNot(message, m1)
        self.assertIsNot(message, m2)
        self.assertEqual(m1.translation(), msgid % {'test1': 'foo', 'test2': 'bar'})
        self.assertEqual(m2.translation(), msgid % {'test1': 'foo2', 'test2': 'bar2'})

    def test_mod_with_none_parameter(self):
        msgid = 'Some string with params: %s'
        message = _message.Message(msgid) % None
        self.assertEqual(msgid % None, message)
        self.assertEqual(msgid % None, message.translation())

    def test_mod_with_missing_parameters(self):
        msgid = 'Some string with params: %s %s'
        test_me = lambda: _message.Message(msgid) % 'just one'
        self.assertRaises(TypeError, test_me)

    def test_mod_with_extra_parameters(self):
        msgid = 'Some string with params: %(param1)s %(param2)s'
        params = {'param1': 'test', 'param2': 'test2', 'param3': 'notinstring'}
        result = _message.Message(msgid) % params
        expected = msgid % params
        self.assertEqual(expected, result)
        self.assertEqual(expected, result.translation())
        self.assertEqual(params.keys(), result.params.keys())

    def test_add_disabled(self):
        msgid = 'A message'
        test_me = lambda: _message.Message(msgid) + ' some string'
        self.assertRaises(TypeError, test_me)

    def test_radd_disabled(self):
        msgid = 'A message'
        test_me = lambda: utils.SomeObject('test') + _message.Message(msgid)
        self.assertRaises(TypeError, test_me)

    @mock.patch('gettext.translation')
    def test_translation(self, mock_translation):
        en_message = 'A message in the default locale'
        es_translation = 'A message in Spanish'
        message = _message.Message(en_message)
        es_translations = {en_message: es_translation}
        translations_map = {'es': es_translations}
        translator = fakes.FakeTranslations.translator(translations_map)
        mock_translation.side_effect = translator
        self.assertEqual(es_translation, message.translation('es'))

    @mock.patch('gettext.translation')
    def test_translate_message_from_unicoded_object(self, mock_translation):
        en_message = 'A message in the default locale'
        es_translation = 'A message in Spanish'
        message = _message.Message(en_message)
        es_translations = {en_message: es_translation}
        translations_map = {'es': es_translations}
        translator = fakes.FakeTranslations.translator(translations_map)
        mock_translation.side_effect = translator
        obj = utils.SomeObject(message)
        unicoded_obj = str(obj)
        self.assertEqual(es_translation, unicoded_obj.translation('es'))

    @mock.patch('gettext.translation')
    def test_translate_multiple_languages(self, mock_translation):
        en_message = 'A message in the default locale'
        es_translation = 'A message in Spanish'
        zh_translation = 'A message in Chinese'
        message = _message.Message(en_message)
        es_translations = {en_message: es_translation}
        zh_translations = {en_message: zh_translation}
        translations_map = {'es': es_translations, 'zh': zh_translations}
        translator = fakes.FakeTranslations.translator(translations_map)
        mock_translation.side_effect = translator
        self.assertEqual(es_translation, message.translation('es'))
        self.assertEqual(zh_translation, message.translation('zh'))
        self.assertEqual(en_message, message.translation(None))
        self.assertEqual(en_message, message.translation('en'))
        self.assertEqual(en_message, message.translation('XX'))

    @mock.patch('gettext.translation')
    def test_translate_message_with_param(self, mock_translation):
        message_with_params = 'A message: %s'
        es_translation = 'A message in Spanish: %s'
        param = 'A Message param'
        translations = {message_with_params: es_translation}
        translator = fakes.FakeTranslations.translator({'es': translations})
        mock_translation.side_effect = translator
        msg = _message.Message(message_with_params)
        msg = msg % param
        default_translation = message_with_params % param
        expected_translation = es_translation % param
        self.assertEqual(expected_translation, msg.translation('es'))
        self.assertEqual(default_translation, msg.translation('XX'))

    @mock.patch('gettext.translation')
    @mock.patch('oslo_i18n._message.LOG')
    def test_translate_message_bad_translation(self, mock_log, mock_translation):
        message_with_params = 'A message: %s'
        es_translation = 'A message in Spanish: %s %s'
        param = 'A Message param'
        translations = {message_with_params: es_translation}
        translator = fakes.FakeTranslations.translator({'es': translations})
        mock_translation.side_effect = translator
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            msg = _message.Message(message_with_params)
            msg = msg % param
            default_translation = message_with_params % param
            self.assertEqual(default_translation, msg.translation('es'))
            self.assertEqual(1, len(w))
            self.assertEqual("Failed to insert replacement values into translated message A message in Spanish: %s %s (Original: 'A message: %s'): not enough arguments for format string", str(w[0].message).replace("u'", "'"))
        mock_log.debug.assert_called_with('Failed to insert replacement values into translated message %s (Original: %r): %s', es_translation, message_with_params, mock.ANY)

    @mock.patch('gettext.translation')
    @mock.patch('locale.getlocale', return_value=('es', ''))
    @mock.patch('oslo_i18n._message.LOG')
    def test_translate_message_bad_default_translation(self, mock_log, mock_locale, mock_translation):
        message_with_params = 'A message: %s'
        es_translation = 'A message in Spanish: %s %s'
        param = 'A Message param'
        translations = {message_with_params: es_translation}
        translator = fakes.FakeTranslations.translator({'es': translations})
        mock_translation.side_effect = translator
        msg = _message.Message(message_with_params)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            msg = msg % param
            self.assertEqual(1, len(w))
            self.assertEqual("Failed to insert replacement values into translated message A message in Spanish: %s %s (Original: 'A message: %s'): not enough arguments for format string", str(w[0].message).replace("u'", "'"))
        mock_log.debug.assert_called_with('Failed to insert replacement values into translated message %s (Original: %r): %s', es_translation, message_with_params, mock.ANY)
        mock_log.reset_mock()
        default_translation = message_with_params % param
        self.assertEqual(default_translation, msg)
        self.assertFalse(mock_log.warning.called)

    @mock.patch('gettext.translation')
    def test_translate_message_with_object_param(self, mock_translation):
        message_with_params = 'A message: %s'
        es_translation = 'A message in Spanish: %s'
        param = 'A Message param'
        param_translation = 'A Message param in Spanish'
        translations = {message_with_params: es_translation, param: param_translation}
        translator = fakes.FakeTranslations.translator({'es': translations})
        mock_translation.side_effect = translator
        msg = _message.Message(message_with_params)
        param_msg = _message.Message(param)
        obj = utils.SomeObject(param_msg)
        msg = msg % obj
        default_translation = message_with_params % param
        expected_translation = es_translation % param_translation
        self.assertEqual(expected_translation, msg.translation('es'))
        self.assertEqual(default_translation, msg.translation('XX'))

    @mock.patch('gettext.translation')
    def test_translate_message_with_param_from_unicoded_obj(self, mock_translation):
        message_with_params = 'A message: %s'
        es_translation = 'A message in Spanish: %s'
        param = 'A Message param'
        translations = {message_with_params: es_translation}
        translator = fakes.FakeTranslations.translator({'es': translations})
        mock_translation.side_effect = translator
        msg = _message.Message(message_with_params)
        msg = msg % param
        default_translation = message_with_params % param
        expected_translation = es_translation % param
        obj = utils.SomeObject(msg)
        unicoded_obj = str(obj)
        self.assertEqual(expected_translation, unicoded_obj.translation('es'))
        self.assertEqual(default_translation, unicoded_obj.translation('XX'))

    @mock.patch('gettext.translation')
    def test_translate_message_with_message_parameter(self, mock_translation):
        message_with_params = 'A message with param: %s'
        es_translation = 'A message with param in Spanish: %s'
        message_param = 'A message param'
        es_param_translation = 'A message param in Spanish'
        translations = {message_with_params: es_translation, message_param: es_param_translation}
        translator = fakes.FakeTranslations.translator({'es': translations})
        mock_translation.side_effect = translator
        msg = _message.Message(message_with_params)
        msg_param = _message.Message(message_param)
        msg = msg % msg_param
        default_translation = message_with_params % message_param
        expected_translation = es_translation % es_param_translation
        self.assertEqual(expected_translation, msg.translation('es'))
        self.assertEqual(default_translation, msg.translation('XX'))

    @mock.patch('gettext.translation')
    def test_translate_message_with_message_parameters(self, mock_translation):
        message_with_params = 'A message with params: %s %s'
        es_translation = 'A message with params in Spanish: %s %s'
        message_param = 'A message param'
        es_param_translation = 'A message param in Spanish'
        another_message_param = 'Another message param'
        another_es_param_translation = 'Another message param in Spanish'
        translations = {message_with_params: es_translation, message_param: es_param_translation, another_message_param: another_es_param_translation}
        translator = fakes.FakeTranslations.translator({'es': translations})
        mock_translation.side_effect = translator
        msg = _message.Message(message_with_params)
        param_1 = _message.Message(message_param)
        param_2 = _message.Message(another_message_param)
        msg = msg % (param_1, param_2)
        default_translation = message_with_params % (message_param, another_message_param)
        expected_translation = es_translation % (es_param_translation, another_es_param_translation)
        self.assertEqual(expected_translation, msg.translation('es'))
        self.assertEqual(default_translation, msg.translation('XX'))

    @mock.patch('gettext.translation')
    def test_translate_message_with_named_parameters(self, mock_translation):
        message_with_params = 'A message with params: %(param)s'
        es_translation = 'A message with params in Spanish: %(param)s'
        message_param = 'A Message param'
        es_param_translation = 'A message param in Spanish'
        translations = {message_with_params: es_translation, message_param: es_param_translation}
        translator = fakes.FakeTranslations.translator({'es': translations})
        mock_translation.side_effect = translator
        msg = _message.Message(message_with_params)
        msg_param = _message.Message(message_param)
        msg = msg % {'param': msg_param}
        default_translation = message_with_params % {'param': message_param}
        expected_translation = es_translation % {'param': es_param_translation}
        self.assertEqual(expected_translation, msg.translation('es'))
        self.assertEqual(default_translation, msg.translation('XX'))

    @mock.patch('locale.getlocale')
    @mock.patch('gettext.translation')
    def test_translate_message_non_default_locale(self, mock_translation, mock_locale):
        message_with_params = 'A message with params: %(param)s'
        es_translation = 'A message with params in Spanish: %(param)s'
        zh_translation = 'A message with params in Chinese: %(param)s'
        fr_translation = 'A message with params in French: %(param)s'
        message_param = 'A Message param'
        es_param_translation = 'A message param in Spanish'
        zh_param_translation = 'A message param in Chinese'
        fr_param_translation = 'A message param in French'
        es_translations = {message_with_params: es_translation, message_param: es_param_translation}
        zh_translations = {message_with_params: zh_translation, message_param: zh_param_translation}
        fr_translations = {message_with_params: fr_translation, message_param: fr_param_translation}
        translator = fakes.FakeTranslations.translator({'es': es_translations, 'zh': zh_translations, 'fr': fr_translations})
        mock_translation.side_effect = translator
        mock_locale.return_value = ('es',)
        msg = _message.Message(message_with_params)
        msg_param = _message.Message(message_param)
        msg = msg % {'param': msg_param}
        es_translation = es_translation % {'param': es_param_translation}
        zh_translation = zh_translation % {'param': zh_param_translation}
        fr_translation = fr_translation % {'param': fr_param_translation}
        self.assertEqual(es_translation, msg)
        self.assertEqual(es_translation, msg.translation())
        self.assertEqual(es_translation, msg.translation('es'))
        self.assertEqual(zh_translation, msg.translation('zh'))
        self.assertEqual(fr_translation, msg.translation('fr'))