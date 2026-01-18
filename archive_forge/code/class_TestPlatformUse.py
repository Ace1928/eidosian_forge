import platform
import re
from io import StringIO
from .. import tests, version, workingtree
from .scenarios import load_tests_apply_scenarios
class TestPlatformUse(tests.TestCase):
    scenarios = [('ascii', dict(_platform='test-platform')), ('unicode', dict(_platform='SchrÃ¶dinger'))]

    def setUp(self):
        super().setUp()
        self.permit_source_tree_branch_repo()

    def test_platform(self):
        out = self.make_utf8_encoded_stringio()
        self.overrideAttr(platform, 'platform', lambda **kwargs: self._platform)
        version.show_version(show_config=False, show_copyright=False, to_file=out)
        expected = '(?m)^  Platform: %s' % self._platform
        expected = expected.encode('utf-8')
        self.assertContainsRe(out.getvalue(), expected)