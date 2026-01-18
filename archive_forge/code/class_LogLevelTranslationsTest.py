from unittest import mock
from oslotest import base as test_base
from oslo_i18n import _factory
class LogLevelTranslationsTest(test_base.BaseTestCase):

    def test_info(self):
        self._test('info')

    def test_warning(self):
        self._test('warning')

    def test_error(self):
        self._test('error')

    def test_critical(self):
        self._test('critical')

    def _test(self, level):
        with mock.patch.object(_factory.TranslatorFactory, '_make_translation_func') as mtf:
            tf = _factory.TranslatorFactory('domain')
            getattr(tf, 'log_%s' % level)
            mtf.assert_called_with('domain-log-%s' % level)