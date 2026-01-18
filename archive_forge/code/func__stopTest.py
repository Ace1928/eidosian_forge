import sys
import threading
import unittest
import gi
from gi.repository import GObject, Gtk    # noqa: E402
from testtools import StreamToExtendedDecorator  # noqa: E402
from subunit import (PROGRESS_POP, PROGRESS_PUSH, PROGRESS_SET,  # noqa: E402
from subunit.progress_model import ProgressModel  # noqa: E402
def _stopTest(self):
    self.progress_model.advance()
    if self.progress_model.width() == 0:
        self.pbar.pulse()
    else:
        pos = self.progress_model.pos()
        width = self.progress_model.width()
        percentage = pos / float(width)
        self.pbar.set_fraction(percentage)