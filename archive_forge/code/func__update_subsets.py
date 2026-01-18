import itertools
import re
from difflib import SequenceMatcher, unified_diff
from pyomo.repn.tests.diffutils import compare_floats, load_baseline
import pyomo.repn.plugins.nl_writer as nl_writer
def _update_subsets(subset, base, test):
    for i, j in zip(*subset):
        if base[i][0] == 'n' and test[j][0] == 'n':
            if compare_floats(base[i][1:], test[j][1:]):
                test[j] = base[i]
        elif compare_floats(base[i], test[j]):
            test[j] = base[i]
        else:
            base_nc = _strip_comment.sub('', base[i])
            test_nc = _strip_comment.sub('', test[j])
            if compare_floats(base_nc, test_nc):
                if len(base_nc) > len(test_nc):
                    test[j] = base[i]
                else:
                    base[i] = test[j]