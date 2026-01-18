from typing import Any, Callable, List, Optional, Tuple, Union
import torch
import torch.utils._pytree as pytree
import torch.utils.dlpack
from torch import Tensor
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import lazy_format_graph_code
from torch._logging import getArtifactLogger
from torch.fx.experimental.proxy_tensor import make_fx
from .functional_utils import assert_functional_graph
from .schemas import AOTConfig, SubclassMeta, ViewAndMutationMeta
from .traced_function_transforms import (
def _create_graph(f, args, *, aot_config: AOTConfig) -> torch.fx.GraphModule:
    with enable_python_dispatcher():
        fx_g = make_fx(f, decomposition_table=aot_config.decompositions)(*args)
    return fx_g