from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
class TestBranchFormatRegistry(tests.TestCase):

    def setUp(self):
        super().setUp()
        self.registry = _mod_branch.BranchFormatRegistry()

    def test_default(self):
        self.assertIs(None, self.registry.get_default())
        format = SampleBranchFormat()
        self.registry.set_default(format)
        self.assertEqual(format, self.registry.get_default())

    def test_register_unregister_format(self):
        format = SampleBranchFormat()
        self.registry.register(format)
        self.assertEqual(format, self.registry.get(b'Sample branch format.'))
        self.registry.remove(format)
        self.assertRaises(KeyError, self.registry.get, b'Sample branch format.')

    def test_get_all(self):
        format = SampleBranchFormat()
        self.assertEqual([], self.registry._get_all())
        self.registry.register(format)
        self.assertEqual([format], self.registry._get_all())

    def test_register_extra(self):
        format = SampleExtraBranchFormat()
        self.assertEqual([], self.registry._get_all())
        self.registry.register_extra(format)
        self.assertEqual([format], self.registry._get_all())

    def test_register_extra_lazy(self):
        self.assertEqual([], self.registry._get_all())
        self.registry.register_extra_lazy('breezy.tests.test_branch', 'SampleExtraBranchFormat')
        formats = self.registry._get_all()
        self.assertEqual(1, len(formats))
        self.assertIsInstance(formats[0], SampleExtraBranchFormat)