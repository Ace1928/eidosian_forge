import collections
import enum
import itertools as it
from typing import DefaultDict, List, Optional, Tuple
from torch.utils.benchmark.utils import common
from torch import tensor as _tensor
def _group_by_label(self, results: List[common.Measurement]):
    grouped_results: DefaultDict[str, List[common.Measurement]] = collections.defaultdict(list)
    for r in results:
        grouped_results[r.label].append(r)
    return grouped_results