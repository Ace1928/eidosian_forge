import os
from breezy.controldir import ControlDir
from breezy.filters import ContentFilter
from breezy.switch import switch
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.workingtree import WorkingTree
def _remove_appended_text(chunks, context=None):
    """Remove the appended text."""
    text = b''.join(chunks)
    if text.endswith(_trailer_string):
        text = text[:-len(_trailer_string)]
    return [text]