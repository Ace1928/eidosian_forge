import os
from breezy.controldir import ControlDir
from breezy.filters import ContentFilter
from breezy.switch import switch
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.workingtree import WorkingTree
def _converter_helper(chunks, fn):
    result = []
    for chunk in chunks:
        result.append(getattr(chunk, fn)())
    return iter(result)