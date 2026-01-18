import inspect
import functools
from enum import Enum
import torch.autograd
def _gen_invalid_iterdatapipe_msg(datapipe):
    return f'This iterator has been invalidated because another iterator has been created from the same IterDataPipe: {_generate_iterdatapipe_msg(datapipe)}\nThis may be caused multiple references to the same IterDataPipe. We recommend using `.fork()` if that is necessary.'