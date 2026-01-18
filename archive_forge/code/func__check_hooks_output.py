from breezy.branch import Branch
from breezy.tests import TestCaseWithTransport
def _check_hooks_output(self, command_output, hooks):
    for hook_type in Branch.hooks:
        s = '\n  '.join(hooks.get(hook_type, ['<no hooks installed>']))
        self.assertTrue('{}:\n    {}'.format(hook_type, s) in command_output)