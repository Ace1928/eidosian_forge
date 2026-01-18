import random
from ._fold_storage import FoldStorage
from ._fold_storage import _FoldFile
def _make_learn_folds(self, fold_size, left_folds):
    """Prepare test sets for folds only for one permutation"""
    count_groups = len(self._groups_ids)
    if count_groups // self._min_folds_count < fold_size:
        raise AttributeError('The size of fold is too big: count_groups: {}, fold_size: {}. Const: {}'.format(count_groups, fold_size, self._min_folds_count))
    permutation = sorted(self._groups_ids)
    self._random.shuffle(permutation)
    result = []
    current_count_folds = min(count_groups // fold_size, left_folds)
    for i in range(current_count_folds):
        result.append(set(permutation[i * fold_size:(i + 1) * fold_size]))
    return result