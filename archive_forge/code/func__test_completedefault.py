import cmd2
from cliff.interactive import InteractiveApp
from cliff.tests import base
def _test_completedefault(self, expecteds, line, begidx):
    command_names = set(['show file', 'show folder', 'show  long', 'list all'])
    app = self.make_interactive_app(False, *command_names)
    observeds = app.completedefault(None, line, begidx, None)
    self.assertEqual(set(expecteds), set(observeds))
    self.assertTrue(set([line[:begidx] + x for x in observeds]) <= command_names)