from typing import TYPE_CHECKING
import torch
from . import allowed_functions
from .eval_frame import DisableContext, innermost_fn, RunOnlyContext
from .exc import IncorrectUsage
def _allow_in_graph_einops():
    import einops
    try:
        from einops._torch_specific import _ops_were_registered_in_torchdynamo
        pass
    except ImportError:
        allow_in_graph(einops.rearrange)
        allow_in_graph(einops.reduce)
        if hasattr(einops, 'repeat'):
            allow_in_graph(einops.repeat)
        if hasattr(einops, 'einsum'):
            allow_in_graph(einops.einsum)
        if hasattr(einops, 'pack'):
            allow_in_graph(einops.pack)
        if hasattr(einops, 'unpack'):
            allow_in_graph(einops.unpack)