from .. import controldir, errors, tests, ui
from .scenarios import load_tests_apply_scenarios
class TestMetaComponentFormatRegistry(tests.TestCase):

    def setUp(self):
        super().setUp()
        self.registry = controldir.ControlComponentFormatRegistry()

    def test_register_unregister_format(self):
        format = SampleComponentFormat()
        self.registry.register(format)
        self.assertEqual(format, self.registry.get(b'Example component format.'))
        self.registry.remove(format)
        self.assertRaises(KeyError, self.registry.get, b'Example component format.')

    def test_get_all(self):
        format = SampleComponentFormat()
        self.assertEqual([], self.registry._get_all())
        self.registry.register(format)
        self.assertEqual([format], self.registry._get_all())

    def test_get_all_modules(self):
        format = SampleComponentFormat()
        self.assertEqual(set(), self.registry._get_all_modules())
        self.registry.register(format)
        self.assertEqual({'breezy.tests.test_controldir'}, self.registry._get_all_modules())

    def test_register_extra(self):
        format = SampleExtraComponentFormat()
        self.assertEqual([], self.registry._get_all())
        self.registry.register_extra(format)
        self.assertEqual([format], self.registry._get_all())

    def test_register_extra_lazy(self):
        self.assertEqual([], self.registry._get_all())
        self.registry.register_extra_lazy('breezy.tests.test_controldir', 'SampleExtraComponentFormat')
        formats = self.registry._get_all()
        self.assertEqual(1, len(formats))
        self.assertIsInstance(formats[0], SampleExtraComponentFormat)