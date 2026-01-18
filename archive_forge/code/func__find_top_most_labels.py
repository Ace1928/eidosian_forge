from functools import reduce
from nltk.parse import load_parser
from nltk.sem.logic import (
from nltk.sem.skolemize import skolemize
def _find_top_most_labels(self):
    """
        Return the set of labels which are not referenced directly as part of
        another formula fragment.  These will be the top-most labels for the
        subtree that they are part of.
        """
    return self._find_top_nodes(self.labels)