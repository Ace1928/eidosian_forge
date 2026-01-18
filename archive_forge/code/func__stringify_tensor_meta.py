import hashlib
import torch
import torch.fx
from typing import Any, Dict, Optional, TYPE_CHECKING
from torch.fx.node import _get_qualified_name, _format_arg
from torch.fx.graph import _parse_stack_trace
from torch.fx.passes.shape_prop import TensorMetadata
from torch.fx._compatibility import compatibility
from itertools import chain
def _stringify_tensor_meta(self, tm: TensorMetadata) -> str:
    result = ''
    if not hasattr(tm, 'dtype'):
        print('tm', tm)
    result += '|' + 'dtype' + '=' + str(tm.dtype) + '\\n'
    result += '|' + 'shape' + '=' + str(tuple(tm.shape)) + '\\n'
    result += '|' + 'requires_grad' + '=' + str(tm.requires_grad) + '\\n'
    result += '|' + 'stride' + '=' + str(tm.stride) + '\\n'
    if tm.is_quantized:
        assert tm.qparams is not None
        assert 'qscheme' in tm.qparams
        qscheme = tm.qparams['qscheme']
        if qscheme in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
            result += '|' + 'q_scale' + '=' + str(tm.qparams['scale']) + '\\n'
            result += '|' + 'q_zero_point' + '=' + str(tm.qparams['zero_point']) + '\\n'
        elif qscheme in {torch.per_channel_affine, torch.per_channel_symmetric, torch.per_channel_affine_float_qparams}:
            result += '|' + 'q_per_channel_scale' + '=' + str(tm.qparams['scale']) + '\\n'
            result += '|' + 'q_per_channel_zero_point' + '=' + str(tm.qparams['zero_point']) + '\\n'
            result += '|' + 'q_per_channel_axis' + '=' + str(tm.qparams['axis']) + '\\n'
        else:
            raise RuntimeError(f'Unsupported qscheme: {qscheme}')
        result += '|' + 'qscheme' + '=' + str(tm.qparams['qscheme']) + '\\n'
    return result