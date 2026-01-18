from typing import Optional, Tuple
import torch
from torch import Tensor
def _symeig(input, eigenvectors=False, upper=True, *, out=None) -> Tuple[Tensor, Tensor]:
    raise RuntimeError("This function was deprecated since version 1.9 and is now removed. The default behavior has changed from using the upper triangular portion of the matrix by default to using the lower triangular portion.\n\nL, _ = torch.symeig(A, upper=upper) should be replaced with:\nL = torch.linalg.eigvalsh(A, UPLO='U' if upper else 'L')\n\nand\n\nL, V = torch.symeig(A, eigenvectors=True) should be replaced with:\nL, V = torch.linalg.eigh(A, UPLO='U' if upper else 'L')")