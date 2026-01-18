from collections import namedtuple
from decimal import Decimal
import numpy as np
from . import backends, blas, helpers, parser, paths, sharing
def _core_contract(operands, contraction_list, backend='auto', evaluate_constants=False, **einsum_kwargs):
    """Inner loop used to perform an actual contraction given the output
    from a ``contract_path(..., einsum_call=True)`` call.
    """
    out_array = einsum_kwargs.pop('out', None)
    specified_out = out_array is not None
    backend = parse_backend(operands, backend)
    no_einsum = not backends.has_einsum(backend)
    for num, contraction in enumerate(contraction_list):
        inds, idx_rm, einsum_str, _, blas_flag = contraction
        if evaluate_constants and any((operands[x] is None for x in inds)):
            return (operands, contraction_list[num:])
        tmp_operands = [operands.pop(x) for x in inds]
        handle_out = specified_out and num + 1 == len(contraction_list)
        if blas_flag and ('EINSUM' not in blas_flag or no_einsum):
            input_str, results_index = einsum_str.split('->')
            input_left, input_right = input_str.split(',')
            tensor_result = ''.join((s for s in input_left + input_right if s not in idx_rm))
            left_pos, right_pos = ([], [])
            for s in idx_rm:
                left_pos.append(input_left.find(s))
                right_pos.append(input_right.find(s))
            new_view = _tensordot(*tmp_operands, axes=(tuple(left_pos), tuple(right_pos)), backend=backend)
            if tensor_result != results_index or handle_out:
                transpose = tuple(map(tensor_result.index, results_index))
                new_view = _transpose(new_view, axes=transpose, backend=backend)
                if handle_out:
                    out_array[:] = new_view
        else:
            if handle_out:
                einsum_kwargs['out'] = out_array
            new_view = _einsum(einsum_str, *tmp_operands, backend=backend, **einsum_kwargs)
        operands.append(new_view)
        del tmp_operands, new_view
    if specified_out:
        return out_array
    else:
        return operands[0]