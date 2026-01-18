from __future__ import annotations
import os
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from traitlets import Bool, Instance, Integer, List, Unicode, default
from nbconvert.utils import _contextlib_chdir
from .latex import LatexExporter
class LatexFailed(IOError):
    """Exception for failed latex run

    Captured latex output is in error.output.
    """

    def __init__(self, output):
        """Initialize the error."""
        self.output = output

    def __unicode__(self):
        """Unicode representation."""
        return 'PDF creating failed, captured latex output:\n%s' % self.output

    def __str__(self):
        """String representation."""
        return self.__unicode__()