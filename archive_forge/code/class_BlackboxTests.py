import sys
import breezy
from breezy import tests
class BlackboxTests(tests.TestCaseWithMemoryTransport):

    def test_zsh_completion(self):
        self.run_bzr('zsh-completion', encoding='utf-8')