import abc
import torch
import itertools
import collections
from torch.nn.modules.module import _addindent
def _get_weight_observer(observer):
    if hasattr(observer, 'activation_post_process'):
        observer = observer.activation_post_process
    return observer