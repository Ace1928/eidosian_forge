import math
from collections.abc import Sequence
from pyomo.common.numeric_types import check_if_numeric_type
def _push_to_discrete_element(self, val, push_to_next_larger_value):
    if not self.step or val in _infinite:
        return val
    else:
        if push_to_next_larger_value:
            _rndFcn = math.ceil if self.step > 0 else math.floor
        else:
            _rndFcn = math.floor if self.step > 0 else math.ceil
        return self.start + self.step * _rndFcn((val - self.start) / float(self.step))