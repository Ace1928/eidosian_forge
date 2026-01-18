import inspect
import functools
from enum import Enum
import torch.autograd
def _generate_iterdatapipe_msg(datapipe, simplify_dp_name: bool=False):
    output_string = f'{datapipe.__class__.__name__}({_generate_input_args_string(datapipe)})'
    if simplify_dp_name:
        output_string = _strip_datapipe_from_name(output_string)
    return output_string