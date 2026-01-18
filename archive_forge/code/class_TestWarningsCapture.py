import warnings
import testtools
import fixtures
class TestWarningsCapture(testtools.TestCase, fixtures.TestWithFixtures):

    def test_capture_reuse(self):
        self.useFixture(fixtures.WarningsFilter())
        warnings.simplefilter('always')
        w = fixtures.WarningsCapture()
        with w:
            warnings.warn('test', DeprecationWarning)
            self.assertEqual(1, len(w.captures))
        with w:
            self.assertEqual([], w.captures)

    def test_capture_message(self):
        self.useFixture(fixtures.WarningsFilter())
        warnings.simplefilter('always')
        w = self.useFixture(fixtures.WarningsCapture())
        warnings.warn('hi', DeprecationWarning)
        self.assertEqual(1, len(w.captures))
        self.assertEqual('hi', str(w.captures[0].message))

    def test_capture_category(self):
        self.useFixture(fixtures.WarningsFilter())
        warnings.simplefilter('always')
        w = self.useFixture(fixtures.WarningsCapture())
        categories = [DeprecationWarning, Warning, UserWarning, SyntaxWarning, RuntimeWarning, UnicodeWarning, FutureWarning]
        for category in categories:
            warnings.warn('test', category)
        self.assertEqual(len(categories), len(w.captures))
        for i, category in enumerate(categories):
            c = w.captures[i]
            self.assertEqual(category, c.category)