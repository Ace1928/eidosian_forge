import builtins
import gettext
import logging
from unittest import mock
from oslotest import base as test_base
from oslo_i18n import _factory
from oslo_i18n import _gettextutils
from oslo_i18n import _lazy
from oslo_i18n import _message
class GettextTest(test_base.BaseTestCase):

    def setUp(self):
        super(GettextTest, self).setUp()
        self._USE_LAZY = _lazy.USE_LAZY
        self.t = _factory.TranslatorFactory('oslo_i18n.test')

    def tearDown(self):
        _lazy.USE_LAZY = self._USE_LAZY
        super(GettextTest, self).tearDown()

    def test_gettext_does_not_blow_up(self):
        LOG.info(self.t.primary('test'))

    def test__gettextutils_install(self):
        _gettextutils.install('blaa')
        _lazy.enable_lazy(False)
        self.assertTrue(isinstance(self.t.primary('A String'), str))
        _gettextutils.install('blaa')
        _lazy.enable_lazy(True)
        self.assertTrue(isinstance(self.t.primary('A Message'), _message.Message))

    def test_gettext_install_looks_up_localedir(self):
        with mock.patch('os.environ.get') as environ_get:
            with mock.patch('gettext.install'):
                environ_get.return_value = '/foo/bar'
                _gettextutils.install('blaa')
                environ_get.assert_has_calls([mock.call('BLAA_LOCALEDIR')])

    def test_gettext_install_updates_builtins(self):
        with mock.patch('os.environ.get') as environ_get:
            with mock.patch('gettext.install'):
                environ_get.return_value = '/foo/bar'
                if '_' in builtins.__dict__:
                    del builtins.__dict__['_']
                _gettextutils.install('blaa')
                self.assertIn('_', builtins.__dict__)

    def test_get_available_languages(self):

        def _mock_gettext_find(domain, localedir=None, languages=None, all=0):
            languages = languages or []
            if domain == 'domain_1':
                if any((x in ['en_GB', 'es_ES', 'fil_PH', 'it'] for x in languages)):
                    return 'translation-file'
            elif domain == 'domain_2':
                if any((x in ['fr_FR', 'zh_HK'] for x in languages)):
                    return 'translation-file'
            return None
        mock_patcher = mock.patch.object(gettext, 'find', _mock_gettext_find)
        mock_patcher.start()
        self.addCleanup(mock_patcher.stop)
        _gettextutils._AVAILABLE_LANGUAGES = {}
        domain_1_languages = _gettextutils.get_available_languages('domain_1')
        domain_2_languages = _gettextutils.get_available_languages('domain_2')
        self.assertEqual('en_US', domain_1_languages[0])
        self.assertEqual('en_US', domain_2_languages[0])
        self.assertEqual(5, len(domain_1_languages), domain_1_languages)
        self.assertEqual({'en_US', 'fil_PH', 'en_GB', 'es_ES', 'it'}, set(domain_1_languages))
        self.assertEqual(3, len(domain_2_languages), domain_2_languages)
        self.assertEqual({'en_US', 'fr_FR', 'zh_HK'}, set(domain_2_languages))
        self.assertEqual(2, len(_gettextutils._AVAILABLE_LANGUAGES))
        unknown_domain_languages = _gettextutils.get_available_languages('huh')
        self.assertEqual(1, len(unknown_domain_languages))
        self.assertIn('en_US', unknown_domain_languages)

    def test_cached_find(self):
        domain = 'my-unique-domain'
        key = (domain, None, None, 0)
        self.assertNotIn(key, _gettextutils._FIND_CACHE)
        gettext.find(domain)
        self.assertIn(key, _gettextutils._FIND_CACHE)
        _gettextutils._FIND_CACHE[key] = 'spoof result'
        self.assertEqual('spoof result', gettext.find(domain))
        _gettextutils._FIND_CACHE.pop(key)