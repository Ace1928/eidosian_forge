from parlai.core.teachers import FbDialogTeacher, FixedDialogTeacher
from .build import build
import copy
import os
def _strip_reader(filename):
    """
    Reads a file, stripping line endings.
    """
    with open(filename) as f:
        for line in f:
            yield line.rstrip()