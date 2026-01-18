import os
import textwrap
import unittest
from gae_ext_runtime import comm
from gae_ext_runtime import ext_runtime
from gae_ext_runtime import testutil
class FakeExecutionEnvironment(ext_runtime.DefaultExecutionEnvironment):

    def CanPrompt(self):
        return True

    def PromptResponse(self, message):
        return 'my_entrypoint'