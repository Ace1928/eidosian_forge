from typing import Any, Dict, TypeVar, Union, List
import torch
class _Shared(Dict[str, Any]):
    """
    ParlAI ``shared`` Structure.

    The `shared` dict that is used to instantiate shared agents in ParlAI,
    e.g. when using batching, distributed training, etc.

    Type is ``TShared``.
    """