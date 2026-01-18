import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, cast
from .backends import get_array_ops
from .config import registry
from .types import FloatsXd, Generator
@registry.optimizers('SGD.v1')
def SGD(learn_rate: FloatOrSeq, *, L2: FloatOrSeq=SGD_DEFAULTS['L2'], grad_clip: FloatOrSeq=SGD_DEFAULTS['grad_clip'], L2_is_weight_decay: bool=cast(bool, SGD_DEFAULTS['L2_is_weight_decay']), use_averages: bool=True):
    return Optimizer(learn_rate, L2=L2, grad_clip=grad_clip, L2_is_weight_decay=L2_is_weight_decay, beta1=0.0, beta2=0.0, use_averages=use_averages)