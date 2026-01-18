from typing import List
from thinc.api import Model, with_flatten_v2
def _memoize_input() -> Model[List[int], List[int]]:
    return Model(name='memoize_input', forward=_memoize_input_forward)