from typing import List
from thinc.api import Model, with_flatten_v2
def _memoize_input_forward(model: Model[List[int], List[int]], X: List[int], is_train: bool):
    model.attrs['last_input'] = X

    def backprop(dY: List[int]):
        return [v + 2 for v in dY]
    return ([v + 1 for v in X], backprop)