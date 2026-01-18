from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.dispatcher as dispatcher
from torchgen.api.translate import translate
from torchgen.api.types import Binding, DispatcherSignature, Expr
from torchgen.context import with_native_function
from torchgen.model import (
from torchgen.utils import concatMap
def add_generated_native_functions(rs: List[NativeFunction], indices: Dict[DispatchKey, Dict[OperatorName, BackendMetadata]]) -> None:
    pre_grouped_native_functions = pre_group_native_functions(rs)
    for d in pre_grouped_native_functions.values():
        has_functional = SchemaKind.functional in d
        has_inplace = SchemaKind.inplace in d
        has_mutable = SchemaKind.mutable in d
        has_out = SchemaKind.out in d
        if has_mutable or has_inplace or has_out or has_functional:
            are_manual = all((f.manual_cpp_binding for f in d.values()))
            has_view_ops = any((f.is_view_op for f in d.values()))
            are_composite_implicit = all((f.has_composite_implicit_autograd_kernel for f in d.values()))
            if are_manual or has_view_ops or are_composite_implicit:
                continue
            if has_out and len(d.values()) == 1:
                if str(d[SchemaKind.out].func.name) not in OUT_OPS_THAT_DONT_GET_GROUPED_PROPERLY:
                    raise AssertionError(f'Found an out= operator that we could not find any other variants of: {str(d[SchemaKind.out].func)}')
                continue
            if has_inplace and str(d[SchemaKind.inplace].func.name) in INPLACE_OPS_THAT_DONT_GET_GROUPED_PROPERLY:
                continue
            base_fn = d[SchemaKind.inplace] if has_inplace else d[SchemaKind.mutable] if has_mutable else d[SchemaKind.out] if has_out else d[SchemaKind.functional]
            base_fn_valid = base_fn.func.kind() == SchemaKind.inplace or any((r.type.is_tensor_like() for r in base_fn.func.returns))
            needs_out = any(('out' in str(op_name) for op_name in base_fn.autogen))
            gets_out_variant = not has_out and base_fn_valid and needs_out
            if not has_out and (not base_fn_valid):
                if str(base_fn.func.name) not in MUTABLE_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT and str(base_fn.func.name) not in FUNCTIONAL_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT:
                    raise AssertionError(f"Found an operator that we could not generate an out= variant for: {str(base_fn.func)}.\nThis type of operators don't have tensor-like return, making it difficult to generate a proper out= variant. If\nout= variant is not needed, please add the function name into FUNCTIONAL_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT list.")
            if gets_out_variant:
                fn, metadata = generate_function(base_fn, SchemaKind.out)
                d[SchemaKind.out] = fn
                BackendIndex.grow_index(indices, metadata)
                rs.append(fn)
            if not has_functional and (has_out or gets_out_variant):
                fn, metadata = generate_function(base_fn, SchemaKind.functional)
                d[SchemaKind.functional] = fn
                BackendIndex.grow_index(indices, metadata)
                rs.append(fn)