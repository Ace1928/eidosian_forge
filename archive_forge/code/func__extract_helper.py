from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import Any, Dict, Iterable, List, Tuple
import numpy as np
import numpy.typing as npt
def _extract_helper(self, src: np.ndarray, offset: int=0):
    start = self.start_idx + offset
    end = self.end_idx + offset
    if self.type == VariableType.SCALAR:
        return src[..., start:end].reshape(-1, *self.dimensions, order='F')
    elif self.type == VariableType.COMPLEX:
        ret = src[..., start:end].reshape(-1, 2, *self.dimensions, order='F')
        ret = ret[:, ::2] + 1j * ret[:, 1::2]
        return ret.squeeze().reshape(-1, *self.dimensions, order='F')
    elif self.type == VariableType.TUPLE:
        out: np.ndarray = np.empty((prod(src.shape[:-1]), prod(self.dimensions)), dtype=object)
        for idx in range(self.num_elts()):
            off = idx * self.elt_size() // self.num_elts()
            elts = [param._extract_helper(src, offset=start + off) for param in self.contents]
            for i in range(elts[0].shape[0]):
                out[i, idx] = tuple((elt[i] for elt in elts))
        return out.reshape(-1, *self.dimensions, order='F')