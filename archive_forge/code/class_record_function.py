from collections import defaultdict
from typing import Any, Dict, List, Optional
from warnings import warn
import torch
import torch.cuda
from torch._C import _get_privateuse1_backend_name
from torch._C._profiler import _ExperimentalConfig
from torch.autograd import (
from torch.autograd.profiler_util import (
from torch.futures import Future
class record_function(_ContextDecorator):
    """Context manager/function decorator that adds a label to a code block/function when running autograd profiler.

    It is useful when tracing the code profile.

    Args:
        name (str): Label assigned to the block of code.
        node_id (int): ID of node, for distributed profiling. Unset in
        non-distributed cases.

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD_PROFILER)
        >>> x = torch.randn((1, 1), requires_grad=True)
        >>> with torch.autograd.profiler.profile() as prof:
        ...     y = x ** 2
        ...     with torch.autograd.profiler.record_function("label-z"): # label the block
        ...         z = y ** 3
        ...     y.backward()
        ...
        >>> # xdoctest: +IGNORE_WANT
        >>> # NOTE: some columns were removed for brevity
        >>> print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        -----------------------------------  ---------------  ---------------  ---------------
        Name                                 Self CPU total %  CPU time avg     Number of Calls
        -----------------------------------  ---------------  ---------------  ---------------
        pow                                  60.77%           47.470us         3
        mul                                  21.73%           25.465us         2
        PowBackward0                         12.03%           121.891us        1
        torch::autograd::AccumulateGrad      2.70%            6.324us          1
        label-z                              2.13%            12.421us         1
        torch::autograd::GraphRoot           0.64%            1.503us          1
        -----------------------------------  ---------------  ---------------  ---------------
        Self CPU time total: 234.344us
        CUDA time total: 0.000us

    """

    def __init__(self, name: str, args: Optional[str]=None):
        self.name: str = name
        self.args: Optional[str] = args
        self.run_callbacks_on_exit: bool = True
        self.record = torch.jit.annotate(Optional['torch.classes.profiler._RecordFunction'], None)

    def __enter__(self):
        self.record = torch.ops.profiler._record_function_enter_new(self.name, self.args)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        if not self.run_callbacks_on_exit:
            return
        record = self.record
        assert record is not None
        if not torch.jit.is_scripting():
            with torch._C.DisableTorchFunctionSubclass():
                torch.ops.profiler._record_function_exit._RecordFunction(record)
        else:
            torch.ops.profiler._record_function_exit(record)

    def _call_end_callbacks_on_future(self, fut: Future[Any]) -> Future[Any]:
        """Use for profiling async calls that return a future.

        Calling this function will extend recording beyond this scope, until the future is
        satisfied. It is useful for profiling the end to end time of asynchronous calls.
        This function should only be called once to attach the callback onto the future, and
        will throw if called multiple times.

        Args:
            fut: (torch._C.Future): future for which to schedule
            callback for.

        Returns:
            A future that completes with the value of the passed in future when
            the profiling callbacks have ran.

        """
        if not self.run_callbacks_on_exit:
            raise RuntimeError('_call_end_callbacks_on_future can only be called once.')
        self.run_callbacks_on_exit = False
        record = self.record
        assert record is not None
        if not torch.jit.is_scripting():
            with torch._C.DisableTorchFunctionSubclass():
                profiled_future = torch.ops.profiler._call_end_callbacks_on_jit_fut._RecordFunction(record, fut)
        else:
            profiled_future = torch.ops.profiler._call_end_callbacks_on_jit_fut(record, fut)
        return profiled_future