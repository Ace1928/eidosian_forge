from typing import Any, Callable, Iterable, Tuple
import torch
from torch import Tensor
from torchaudio._internal import module_utils as _mod_utils
def _convert_method_output_to_tensor(file_or_fd: Any, fn: Callable, convert_contiguous: bool=False) -> Iterable[Tuple[str, Tensor]]:
    """Takes a method invokes it. The output is converted to a tensor.

    Args:
        file_or_fd (str/FileDescriptor): File name or file descriptor
        fn (Callable): Function that has the signature (file name/descriptor) and converts it to
            Iterable[Tuple[str, Tensor]].
        convert_contiguous (bool, optional): Determines whether the array should be converted into a
            contiguous layout. (Default: ``False``)

    Returns:
        Iterable[Tuple[str, Tensor]]: The string is the key and the tensor is vec/mat
    """
    for key, np_arr in fn(file_or_fd):
        if convert_contiguous:
            np_arr = np.ascontiguousarray(np_arr)
        yield (key, torch.from_numpy(np_arr))