from __future__ import annotations
from typing import Dict
from torch import _C
Extra context for symbolic functions.

    Args:
        params_dict (Dict[str, _C.IValue]): Mapping from graph initializer name to IValue.
        env (Dict[_C.Value, _C.Value]): Mapping from Torch domain graph Value to ONNX domain graph Value.
        cur_node (_C.Node): Current node being converted to ONNX domain.
        onnx_block (_C.Block): Current ONNX block that converted nodes are being appended to.
    