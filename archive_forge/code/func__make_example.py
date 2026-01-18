import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
import wandb
def _make_example(data: Any) -> Optional[Union[Dict, Sequence, Any]]:
    """Used to make an example input, target, or output."""
    example: Optional[Union[Dict, Sequence, Any]]
    if isinstance(data, dict):
        example = {}
        for key in data:
            example[key] = data[key][0]
    elif hasattr(data, '__len__'):
        example = data[0]
    else:
        example = None
    return example