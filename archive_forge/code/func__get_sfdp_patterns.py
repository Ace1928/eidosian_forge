import functools
import inspect
import logging
import math
import torch
from ..._dynamo.utils import counters
from ..pattern_matcher import (
def _get_sfdp_patterns():
    from .joint_graph import patterns
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    g_inp = functools.partial(torch.empty, (2, 4, 8, 16), device=device, requires_grad=True)
    b_inp = functools.partial(torch.empty, (1, 1, 8, 8), device=device)
    c_inp = functools.partial(torch.tensor, 2.0, device=device)
    d = {'dropout_p': 0.113377}
    g_3d_inp = functools.partial(torch.empty, (1024, 128, 128), device=device, requires_grad=True)
    for dtype in [torch.float, torch.half]:
        g = functools.partial(g_inp, dtype=dtype)
        b = functools.partial(b_inp, dtype=dtype)
        c = functools.partial(c_inp, dtype=dtype)
        g_3d = functools.partial(g_3d_inp, dtype=dtype)
        for pattern, replacement, args, workaround, extra_check in [(_sfdp_pattern_1, _sfdp_replacement_1, [g(), g(), g(), c()], {}, _sfdp_scale_factor_check(aten.div.Tensor)), (_sfdp_pattern_2, _sfdp_replacement_2, [g(), g(), g(), c()], {}, _sfdp_scale_factor_check(aten.mul.Tensor)), (_sfdp_pattern_3, _sfdp_replacement_3, [g(), g(), g(), c()], d, _sfdp_scale_factor_check(aten.div.Tensor)), (_sfdp_pattern_4, _sfdp_replacement_4, [g(), g(), g(), c()], d, _sfdp_scale_factor_check(aten.mul.Tensor)), (_sfdp_pattern_5, _sfdp_replacement_5, [g(), g(), g(), b()], {}, _sfdp_params_check), (_sfdp_pattern_6, _sfdp_replacement_6, [g(), g(), g(), b()], d, _sfdp_params_check), (_sfdp_pattern_7, _sfdp_replacement_7, [g(), g(), g()], d, _sfdp_params_check), (_sfdp_pattern_8, _sfdp_replacement_8, [g(), g(), g()], {}, _sfdp_params_check), (_sfdp_pattern_9, _sfdp_replacement_9, [g(), g(), g()], d, _sfdp_params_check), (_sfdp_pattern_10, _sfdp_replacement_10, [g(), g(), g()], {}, _sfdp_params_check), (_sfdp_pattern_11, _sfdp_replacement_11, [g(), g(), g(), c()], {}, _sfdp_scale_factor_check(aten.div.Tensor)), (_sfdp_pattern_12, _sfdp_replacement_12, [g(), g(), g(), c()], d, _sfdp_scale_factor_check(aten.div.Tensor)), (_sfdp_pattern_13, _sfdp_replacement_13, [g_3d(), g_3d(), g_3d()], d, _sfdp_params_check)]:
            assert isinstance(workaround, dict)
            name = pattern.__name__
            training_name = f'{name}_training' if dtype == torch.float else f'{name}_training_half'
            yield (training_name, {'search_fn': pattern, 'replace_fn': replacement, 'example_inputs': args, 'trace_fn': joint_fwd_bwd, 'pass_dicts': patterns, 'extra_check': extra_check, 'scalar_workaround': workaround})
            if workaround:
                assert len(workaround) == 1 and 'dropout_p' in workaround
                pattern = partialize_and_update_signature(pattern, dropout_p=0.0)
                replacement = partialize_and_update_signature(replacement, dropout_p=0.0)
                workaround = {}
            inference_name = f'{name}_inference' if dtype == torch.float else f'{name}_inference_half'
            yield (inference_name, {'search_fn': pattern, 'replace_fn': replacement, 'example_inputs': args, 'trace_fn': fwd_only, 'pass_dicts': patterns, 'extra_check': extra_check, 'scalar_workaround': workaround})