from typing import Any, TypeVar, Optional, Tuple, List, NamedTuple, Union, Sequence, Dict, Callable
import textwrap
import torch
from torch._C import TupleType, ListType
from torch.jit._recursive import wrap_cpp_module
def _inflate_expr(arg: T, ref: str, inflate_helper_fn_name: str, skip_size_check: bool=False) -> Tuple[Union[T, torch.Tensor], str, Optional[str]]:
    if isinstance(arg, InflatableArg):
        if arg.fmt_fn:
            if arg.fmt not in ['{}', '']:
                raise Exception(f"Bundled input argument at position '{ref}' has both arg.fmt_fn => \n{arg.fmt_fn} \n and arg.fmt  => {arg.fmt}. Please choose `arg.fmt` if the deflater is straightforward or `arg.fmt_fn` if you need a function.")
            helper_definition = arg.fmt_fn.format(inflate_helper_fn_name)
            expr = f'self.{inflate_helper_fn_name}({ref})'
            return (arg.value, expr, helper_definition)
        else:
            return (arg.value, arg.fmt.format(ref), None)
    if isinstance(arg, torch.Tensor):
        if arg._typed_storage().size() <= MAX_RAW_TENSOR_SIZE or skip_size_check:
            return (arg, ref, None)
        if arg.is_contiguous() and arg.numel() <= MAX_RAW_TENSOR_SIZE:
            return (arg.clone(), ref, None)
        for fmt in [torch.contiguous_format, torch.channels_last]:
            if arg.is_contiguous(memory_format=fmt) and (arg == arg.flatten()[0]).all().item():
                return (arg.flatten()[0].clone().expand(*arg.size()), f'{ref}.contiguous(memory_format={fmt})', None)
        raise Exception(f"Bundled input argument at position '{ref}' is a tensor with storage size {arg._typed_storage().size()}. You probably don't want to bundle this as an input. ")
    else:
        return (arg, ref, None)