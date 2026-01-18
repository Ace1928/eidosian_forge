from __future__ import print_function
from patsy.util import no_pickling
def _subsets_sorted(tupl):

    def helper(seq):
        if not seq:
            yield ()
        else:
            obj = seq[0]
            for subset in _subsets_sorted(seq[1:]):
                yield subset
                yield ((obj,) + subset)
    expanded = list(enumerate(tupl))
    expanded_subsets = list(helper(expanded))
    expanded_subsets.sort()
    expanded_subsets.sort(key=len)
    for subset in expanded_subsets:
        yield tuple([obj for idx, obj in subset])