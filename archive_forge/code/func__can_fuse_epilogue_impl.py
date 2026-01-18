import logging
from typing import cast, List
from ...._dynamo.utils import counters
from ... import config, ir
from ...codecache import code_hash, get_path
from ...ir import ComputedBuffer, CUDATemplateBuffer, Pointwise
from ...scheduler import (
from ...utils import get_fused_kernel_name, get_kernel_metadata, sympy_product
from ...virtualized import V
from ..common import IndentedBuffer
from .cutlass_epilogue_gen import CUTLASSEVTOpNotImplementedError
def _can_fuse_epilogue_impl(self, cuda_template_buffer: CUDATemplateBuffer, epilogue_nodes: List[ir.IRNode], additional_node: ir.IRNode) -> bool:
    """
        Check if the given node can be fused with the epilogue. At the moment, Kernels
        support fusion with Pointwise operations, wrapped in (named) ComputedBuffer nodes.

        Args:
            cuda_template_buffer : A CUDATemplateBuffer object representing the CUDA template and it's result buffer
            epilogue_nodes : List[ir.Buffer]: The list of already fused epilogue nodes.
            additional_node: The ir.Buffer node to be checked if it can be fused with the epilogue.
        Returns:
        - bool: True if the given node can be fused with the epilogue, False otherwise.

        """
    if not isinstance(cuda_template_buffer, CUDATemplateBuffer):
        return False
    if not cuda_template_buffer.template.can_fuse_epilogue:
        return False
    if not isinstance(additional_node, ComputedBuffer):
        return False
    if not isinstance(additional_node.data, Pointwise):
        return False
    node_name = additional_node.get_computed_buffer_name()
    if node_name is None:
        return False
    if len(epilogue_nodes) == 0:
        if cuda_template_buffer.name not in additional_node.get_read_names():
            return False
    else:
        last_epilogue_node = epilogue_nodes[-1]
        assert isinstance(last_epilogue_node, ir.ComputedBuffer)
        last_epilogue_name = last_epilogue_node.name if last_epilogue_node.name is not None else last_epilogue_node.data.name
        if last_epilogue_name not in additional_node.get_read_names():
            return False
    if additional_node.layout != cuda_template_buffer.layout:
        return False
    try:
        from torch._inductor.codegen.cuda.cutlass_epilogue_gen import CutlassEVTEpilogueArgumentFormatter, CutlassEVTEpilogueTypeFormatter
        CutlassEVTEpilogueTypeFormatter.ir_to_evt_string(cast(str, cuda_template_buffer.name), 'anything', [additional_node])
        CutlassEVTEpilogueArgumentFormatter.ir_to_evt_argument_string(cast(str, cuda_template_buffer.name), [additional_node])
    except CUTLASSEVTOpNotImplementedError as e:
        not_implemented_op = str(e)
        if not_implemented_op.startswith('_op_'):
            not_implemented_op = not_implemented_op[4:]
            log.warning(f'Cannot fuse epilogue node {additional_node} into {cuda_template_buffer.name}, likely due to unsupported operation: {not_implemented_op}')
            return False
        else:
            log.warning(f'Cannot fuse epilogue node {additional_node} into {cuda_template_buffer.name}. Reason: {not_implemented_op}')
            return False
    return True