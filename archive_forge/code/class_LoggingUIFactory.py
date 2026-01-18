import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
class LoggingUIFactory(breezy.ui.SilentUIFactory):

    def __init__(self):
        self.prompts = []

    def get_boolean(self, prompt):
        self.prompts.append(('boolean', prompt))
        return True