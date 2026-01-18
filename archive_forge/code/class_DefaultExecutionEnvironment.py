from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import logging
import os
import subprocess
import sys
import threading
from . import comm
import ruamel.yaml as yaml
from six.moves import input
class DefaultExecutionEnvironment(ExecutionEnvironment):
    """Standard implementation of the ExecutionEnvironment."""

    def GetPythonExecutable(self):
        return sys.executable

    def CanPrompt(self):
        return sys.stdin.isatty()

    def PromptResponse(self, message):
        sys.stdout.write(message)
        sys.stdout.flush()
        return input('> ')

    def Print(self, message):
        print(message)