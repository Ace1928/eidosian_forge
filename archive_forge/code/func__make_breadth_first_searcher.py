import time
from . import debug, errors, osutils, revision, trace
def _make_breadth_first_searcher(self, revisions):
    return _BreadthFirstSearcher(revisions, self)