from typing import List
import torch
class _dispatch_dtypes(tuple):

    def __add__(self, other):
        assert isinstance(other, tuple)
        return _dispatch_dtypes(tuple.__add__(self, other))