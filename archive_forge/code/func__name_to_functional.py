import functools
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union
import torch
from torchvision import tv_tensors
def _name_to_functional(name):
    import torchvision.transforms.v2.functional
    try:
        return getattr(torchvision.transforms.v2.functional, name)
    except AttributeError:
        raise ValueError(f"Could not find functional with name '{name}' in torchvision.transforms.v2.functional.") from None