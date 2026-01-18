import cmd2
from cliff.interactive import InteractiveApp
from cliff.tests import base
def _test_completenames(self, expecteds, prefix):
    app = self.make_interactive_app(False, 'hips', 'hippo', 'nonmatching')
    self.assertEqual(set(app.completenames(prefix, '', 0, 1)), set(expecteds))