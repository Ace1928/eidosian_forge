import torch
from torch.utils._pytree import tree_map
from typing import Iterator, List, Optional
import logging
import contextlib
import itertools
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.weak import WeakTensorKeyDictionary
import functools
from torch._C._profiler import gather_traceback, symbolize_tracebacks
class LoggingTensorHandler(logging.Handler):

    def __init__(self, log_list: List[str], use_shortid_for_all_tensors: bool, with_type: bool, tracebacks_list: Optional[List]) -> None:
        logging.Handler.__init__(self)
        self.log_list = log_list
        self.use_shortid_for_all_tensors = use_shortid_for_all_tensors
        self.tracebacks_list = tracebacks_list
        self.memo = WeakTensorKeyDictionary()
        self.next_id = 0
        self.with_type = with_type

    def _shortid(self, t: torch.Tensor) -> int:
        if t not in self.memo:
            self.memo[t] = self.next_id
            self.next_id += 1
        return self.memo[t]

    def _fmt(self, a: object, with_type: bool=False) -> str:
        cond_cls = torch.Tensor if self.use_shortid_for_all_tensors else LoggingTensor
        if isinstance(a, cond_cls):
            maybe_type = ''
            if with_type and self.with_type:
                maybe_type = f': {_dtype_abbrs[a.dtype]}[{', '.join(map(str, a.shape))}]'
            x = f'${self._shortid(a)}{maybe_type}'
            return x
        else:
            return repr(a)

    def emit(self, record):
        fmt_args = ', '.join(itertools.chain((str(tree_map(self._fmt, a)) for a in record.args[0]), (f'{k}={str(tree_map(self._fmt, v))}' for k, v in record.args[1].items())))
        fmt_rets = tree_map(functools.partial(self._fmt, with_type=True), record.args[2])
        self.log_list.append(f'{fmt_rets} = {record.msg}({fmt_args})')
        if self.tracebacks_list is not None:
            self.tracebacks_list.append(record.traceback)