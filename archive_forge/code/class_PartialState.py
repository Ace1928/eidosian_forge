from __future__ import annotations
import logging
import math
import os
import threading
import warnings
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Optional
import torch
from .utils import (
from .utils.dataclasses import SageMakerDistributedType
class PartialState:
    """
    Singleton class that has information about the current training environment and functions to help with process
    control. Designed to be used when only process control and device execution states are needed. Does *not* need to
    be initialized from `Accelerator`.

    **Available attributes:**

        - **device** (`torch.device`) -- The device to use.
        - **distributed_type** ([`~accelerate.state.DistributedType`]) -- The type of distributed environment currently
          in use.
        - **local_process_index** (`int`) -- The index of the current process on the current server.
        - **mixed_precision** (`str`) -- Whether or not the current script will use mixed precision, and if so the type
          of mixed precision being performed. (Choose from 'no','fp16','bf16 or 'fp8').
        - **num_processes** (`int`) -- The number of processes currently launched in parallel.
        - **process_index** (`int`) -- The index of the current process.
        - **is_last_process** (`bool`) -- Whether or not the current process is the last one.
        - **is_main_process** (`bool`) -- Whether or not the current process is the main one.
        - **is_local_main_process** (`bool`) -- Whether or not the current process is the main one on the local node.
        - **debug** (`bool`) -- Whether or not the current script is being run in debug mode.
    """
    _shared_state = SharedDict()

    def __init__(self, cpu: bool=False, **kwargs):
        self.__dict__ = self._shared_state
        if not self.initialized:
            self._cpu = cpu
            self.backend = None
            env_device = os.environ.get('ACCELERATE_TORCH_DEVICE', None)
            self.device = torch.device(env_device) if env_device is not None else None
            self.debug = parse_flag_from_env('ACCELERATE_DEBUG_MODE')
            use_sagemaker_dp = kwargs.pop('_use_sagemaker_dp', None)
            if use_sagemaker_dp is None:
                use_sagemaker_dp = os.environ.get('ACCELERATE_USE_SAGEMAKER', 'false') == 'true' and os.environ.get('ACCELERATE_SAGEMAKER_DISTRIBUTED_TYPE') != SageMakerDistributedType.NO
            if use_sagemaker_dp and (not cpu):
                if os.environ.get('ACCELERATE_SAGEMAKER_DISTRIBUTED_TYPE') == SageMakerDistributedType.DATA_PARALLEL or use_sagemaker_dp:
                    self.distributed_type = DistributedType.MULTI_GPU
                    import smdistributed.dataparallel.torch.torch_smddp
                    if not torch.distributed.is_initialized():
                        torch.distributed.init_process_group(backend='smddp')
                    self.backend = 'smddp'
                    self.num_processes = torch.distributed.get_world_size()
                    self.process_index = torch.distributed.get_rank()
                    self.local_process_index = int(os.environ.get('LOCAL_RANK', -1))
                    if self.device is None:
                        self.device = torch.device('cuda', self.local_process_index)
                    torch.cuda.set_device(self.device)
            elif is_torch_xla_available() and (not cpu):
                self.distributed_type = DistributedType.XLA
                self.device = xm.xla_device()
                xm.set_replication(self.device, [self.device])
                self.num_processes = xm.xrt_world_size()
                self.process_index = xm.get_ordinal()
                if is_torch_xla_available(check_is_tpu=True):
                    self.local_process_index = xm.get_local_ordinal()
                else:
                    self.local_process_index = int(os.environ.get('LOCAL_RANK', -1))
            elif os.environ.get('ACCELERATE_USE_DEEPSPEED', 'false') == 'true' and int(os.environ.get('LOCAL_RANK', -1)) != -1 and (not cpu):
                assert is_deepspeed_available(), 'DeepSpeed is not available => install it using `pip3 install deepspeed` or build it from source'
                self.distributed_type = DistributedType.DEEPSPEED
                if not torch.distributed.is_initialized():
                    from deepspeed import comm as dist
                    kwargs.pop('backend', None)
                    if is_xpu_available and is_ccl_available():
                        self.backend = 'ccl'
                        os.environ['CCL_PROCESS_LAUNCHER'] = 'none'
                        os.environ['CCL_LOCAL_SIZE'] = os.environ.get('LOCAL_WORLD_SIZE', '1')
                        os.environ['CCL_LOCAL_RANK'] = os.environ.get('LOCAL_RANK', '0')
                    elif is_npu_available():
                        self.backend = 'hccl'
                    else:
                        self.backend = 'nccl'
                    dist.init_distributed(dist_backend=self.backend, auto_mpi_discovery=False, **kwargs)
                self.num_processes = torch.distributed.get_world_size()
                self.process_index = torch.distributed.get_rank()
                self.local_process_index = int(os.environ.get('LOCAL_RANK', -1))
                if self.device is None:
                    if is_xpu_available():
                        self.device = torch.device('xpu', self.local_process_index)
                        if self.device is not None:
                            torch.xpu.set_device(self.device)
                    elif is_npu_available():
                        self.device = torch.device('npu', self.local_process_index)
                        if self.device is not None:
                            torch.npu.set_device(self.device)
                    else:
                        self.device = torch.device('cuda', self.local_process_index)
                        if self.device is not None:
                            torch.cuda.set_device(self.device)
                if self.device.type == 'cuda' and (not check_cuda_p2p_ib_support()):
                    if 'NCCL_P2P_DISABLE' not in os.environ or 'NCCL_IB_DISABLE' not in os.environ:
                        raise NotImplementedError('Using RTX 4000 series doesn\'t support faster communication broadband via P2P or IB. Please set `NCCL_P2P_DISABLE="1"` and `NCCL_IB_DISABLE="1" or use `accelerate launch` which will do this automatically.')
                self._mixed_precision = 'no'
            elif int(os.environ.get('LOCAL_RANK', -1)) != -1 and (not cpu) and torch.cuda.is_available():
                self.distributed_type = DistributedType.MULTI_GPU
                if not torch.distributed.is_initialized():
                    self.backend = kwargs.pop('backend', 'nccl')
                    if self.backend is None:
                        self.backend = 'nccl'
                    torch.distributed.init_process_group(backend=self.backend, **kwargs)
                if not check_cuda_p2p_ib_support():
                    if 'NCCL_P2P_DISABLE' not in os.environ or 'NCCL_IB_DISABLE' not in os.environ:
                        raise NotImplementedError('Using RTX 4000 series doesn\'t support faster communication broadband via P2P or IB. Please set `NCCL_P2P_DISABLE="1"` and `NCCL_IB_DISABLE="1" or use `accelerate launch` which will do this automatically.')
                self.num_processes = torch.distributed.get_world_size()
                self.process_index = torch.distributed.get_rank()
                self.local_process_index = int(os.environ.get('LOCAL_RANK', -1))
                if self.device is None:
                    self.device = torch.device('cuda', self.local_process_index)
                torch.cuda.set_device(self.device)
            elif is_npu_available() and (not cpu) and (int(os.environ.get('LOCAL_RANK', -1)) != -1):
                self.distributed_type = DistributedType.MULTI_NPU
                if not torch.distributed.is_initialized():
                    kwargs.pop('backend', None)
                    self.backend = 'hccl'
                    torch.distributed.init_process_group(backend=self.backend, **kwargs)
                self.num_processes = torch.distributed.get_world_size()
                self.process_index = torch.distributed.get_rank()
                self.local_process_index = int(os.environ.get('LOCAL_RANK', -1))
                if self.device is None:
                    self.device = torch.device('npu', self.local_process_index)
                torch.npu.set_device(self.device)
            elif get_int_from_env(['PMI_SIZE', 'OMPI_COMM_WORLD_SIZE', 'MV2_COMM_WORLD_SIZE', 'WORLD_SIZE'], 1) > 1 or int(os.environ.get('LOCAL_RANK', -1)) != -1:
                if not cpu and is_xpu_available():
                    self.distributed_type = DistributedType.MULTI_XPU
                else:
                    self.distributed_type = DistributedType.MULTI_CPU
                if is_ccl_available() and (get_int_from_env(['CCL_WORKER_COUNT'], 0) > 0 or self.distributed_type == DistributedType.MULTI_XPU):
                    if get_ccl_version() >= '1.12':
                        import oneccl_bindings_for_pytorch
                    else:
                        import torch_ccl
                    backend = 'ccl'
                elif torch.distributed.is_mpi_available():
                    backend = 'mpi'
                else:
                    backend = 'gloo'
                rank = get_int_from_env(['RANK', 'PMI_RANK', 'OMPI_COMM_WORLD_RANK', 'MV2_COMM_WORLD_RANK'], 0)
                size = get_int_from_env(['WORLD_SIZE', 'PMI_SIZE', 'OMPI_COMM_WORLD_SIZE', 'MV2_COMM_WORLD_SIZE'], 1)
                local_rank = get_int_from_env(['LOCAL_RANK', 'MPI_LOCALRANKID', 'OMPI_COMM_WORLD_LOCAL_RANK', 'MV2_COMM_WORLD_LOCAL_RANK'], 0)
                local_size = get_int_from_env(['LOCAL_WORLD_SIZE', 'MPI_LOCALNRANKS', 'OMPI_COMM_WORLD_LOCAL_SIZE', 'MV2_COMM_WORLD_LOCAL_SIZE'], 1)
                self.local_process_index = local_rank
                os.environ['RANK'] = str(rank)
                os.environ['WORLD_SIZE'] = str(size)
                os.environ['LOCAL_RANK'] = str(local_rank)
                os.environ['LOCAL_WORLD_SIZE'] = str(local_size)
                if backend == 'ccl' and self.distributed_type == DistributedType.MULTI_XPU:
                    os.environ['CCL_PROCESS_LAUNCHER'] = 'none'
                    os.environ['CCL_LOCAL_SIZE'] = str(local_size)
                    os.environ['CCL_LOCAL_RANK'] = str(local_rank)
                if not os.environ.get('MASTER_PORT', None):
                    os.environ['MASTER_PORT'] = '29500'
                if not os.environ.get('MASTER_ADDR', None):
                    if local_size != size and backend != 'mpi':
                        raise ValueError("Looks like distributed multinode run but MASTER_ADDR env not set, please try exporting rank 0's hostname as MASTER_ADDR")
                if self.distributed_type == DistributedType.MULTI_CPU and get_int_from_env(['OMP_NUM_THREADS', 'MKL_NUM_THREADS'], 0) == 0:
                    import psutil
                    num_cpu_threads_per_process = int(psutil.cpu_count(logical=False) / local_size)
                    if num_cpu_threads_per_process == 0:
                        num_cpu_threads_per_process = 1
                    torch.set_num_threads(num_cpu_threads_per_process)
                    warnings.warn(f'OMP_NUM_THREADS/MKL_NUM_THREADS unset, we set it at {num_cpu_threads_per_process} to improve oob performance.')
                if not torch.distributed.is_initialized():
                    kwargs.pop('backend', None)
                    self.backend = backend
                    torch.distributed.init_process_group(self.backend, rank=rank, world_size=size, **kwargs)
                self.num_processes = torch.distributed.get_world_size()
                self.process_index = torch.distributed.get_rank()
                if cpu:
                    self.device = torch.device('cpu')
                elif is_xpu_available():
                    self.device = torch.device('xpu', self.local_process_index)
                    torch.xpu.set_device(self.device)
                else:
                    self.device = self.default_device
            else:
                self.distributed_type = DistributedType.NO if os.environ.get('ACCELERATE_USE_DEEPSPEED', 'false') == 'false' else DistributedType.DEEPSPEED
                self.num_processes = 1
                self.process_index = self.local_process_index = 0
                if self.device is None:
                    self.device = torch.device('cpu') if cpu else self.default_device
        self.fork_launched = parse_flag_from_env('FORK_LAUNCHED', 0)

    def __repr__(self) -> str:
        return f'Distributed environment: {self.distributed_type}{('  Backend: ' + self.backend if self.backend else '')}\nNum processes: {self.num_processes}\nProcess index: {self.process_index}\nLocal process index: {self.local_process_index}\nDevice: {self.device}\n'

    @staticmethod
    def _reset_state():
        """Resets `_shared_state`, is used internally and should not be called"""
        PartialState._shared_state.clear()

    @property
    def initialized(self) -> bool:
        """Returns whether the `PartialState` has been initialized"""
        return self._shared_state != {}

    @property
    def use_distributed(self):
        """
        Whether the Accelerator is configured for distributed training
        """
        return self.distributed_type != DistributedType.NO and self.num_processes > 1

    @property
    def is_last_process(self) -> bool:
        """Returns whether the current process is the last one"""
        return self.process_index == self.num_processes - 1

    @property
    def is_main_process(self) -> bool:
        """Returns whether the current process is the main process"""
        return self.process_index == 0 if self.distributed_type != DistributedType.MEGATRON_LM else self.is_last_process

    @property
    def is_local_main_process(self) -> bool:
        """Returns whether the current process is the main process on the local node"""
        return self.local_process_index == 0 if self.distributed_type != DistributedType.MEGATRON_LM else self.is_last_process

    def wait_for_everyone(self):
        """
        Will stop the execution of the current process until every other process has reached that point (so this does
        nothing when the script is only run in one process). Useful to do before saving a model.

        Example:

        ```python
        >>> # Assuming two GPU processes
        >>> import time
        >>> from accelerate.state import PartialState

        >>> state = PartialState()
        >>> if state.is_main_process:
        ...     time.sleep(2)
        >>> else:
        ...     print("I'm waiting for the main process to finish its sleep...")
        >>> state.wait_for_everyone()
        >>> # Should print on every process at the same time
        >>> print("Everyone is here")
        ```
        """
        if self.distributed_type in (DistributedType.MULTI_GPU, DistributedType.MULTI_NPU, DistributedType.MULTI_XPU, DistributedType.MULTI_CPU, DistributedType.DEEPSPEED, DistributedType.FSDP):
            torch.distributed.barrier()
        elif self.distributed_type == DistributedType.XLA:
            xm.rendezvous('accelerate.utils.wait_for_everyone')

    def _goes_first(self, is_main: bool):
        if not is_main:
            self.wait_for_everyone()
        yield
        if is_main:
            self.wait_for_everyone()

    @contextmanager
    def split_between_processes(self, inputs: list | tuple | dict | torch.Tensor, apply_padding: bool=False):
        """
        Splits `input` between `self.num_processes` quickly and can be then used on that process. Useful when doing
        distributed inference, such as with different prompts.

        Note that when using a `dict`, all keys need to have the same number of elements.

        Args:
            inputs (`list`, `tuple`, `torch.Tensor`, or `dict` of `list`/`tuple`/`torch.Tensor`):
                The input to split between processes.
            apply_padding (`bool`, `optional`, defaults to `False`):
                Whether to apply padding by repeating the last element of the input so that all processes have the same
                number of elements. Useful when trying to perform actions such as `gather()` on the outputs or passing
                in less inputs than there are processes. If so, just remember to drop the padded elements afterwards.


        Example:

        ```python
        # Assume there are two processes
        from accelerate import PartialState

        state = PartialState()
        with state.split_between_processes(["A", "B", "C"]) as inputs:
            print(inputs)
        # Process 0
        ["A", "B"]
        # Process 1
        ["C"]

        with state.split_between_processes(["A", "B", "C"], apply_padding=True) as inputs:
            print(inputs)
        # Process 0
        ["A", "B"]
        # Process 1
        ["C", "C"]
        ```
        """
        if self.num_processes == 1:
            yield inputs
            return
        length = len(inputs)
        if isinstance(inputs, dict):
            length = len(inputs[list(inputs.keys())[0]])
            if not all((len(v) == length for v in inputs.values())):
                raise ValueError('All values in the dictionary must have the same length')
        num_samples_per_process = math.ceil(length / self.num_processes)
        start_index = self.process_index * num_samples_per_process
        end_index = start_index + num_samples_per_process
        if len(inputs) % self.num_processes != 0 and self.process_index == self.num_processes - 1:
            end_index = length

        def _split_values(inputs, start_index, end_index):
            if isinstance(inputs, (list, tuple, torch.Tensor)):
                if start_index >= len(inputs):
                    result = inputs[-1:]
                else:
                    result = inputs[start_index:end_index]
                if apply_padding:
                    if isinstance(result, torch.Tensor):
                        from accelerate.utils import pad_across_processes, send_to_device
                        tensorized_result = send_to_device(result, self.device)
                        result = pad_across_processes(tensorized_result, pad_index=inputs[-1])
                    else:
                        result += [result[-1]] * (num_samples_per_process - len(result))
                return result
            elif isinstance(inputs, dict):
                for key in inputs.keys():
                    inputs[key] = _split_values(inputs[key], start_index, end_index)
                return inputs
            else:
                return inputs
        yield _split_values(inputs, start_index, end_index)

    @contextmanager
    def main_process_first(self):
        """
        Lets the main process go first inside a with block.

        The other processes will enter the with block after the main process exits.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> with accelerator.main_process_first():
        ...     # This will be printed first by process 0 then in a seemingly
        ...     # random order by the other processes.
        ...     print(f"This will be printed by process {accelerator.process_index}")
        ```
        """
        yield from self._goes_first(self.is_main_process)

    @contextmanager
    def local_main_process_first(self):
        """
        Lets the local main process go inside a with block.

        The other processes will enter the with block after the main process exits.

        Example:

        ```python
        >>> from accelerate.state import PartialState

        >>> state = PartialState()
        >>> with state.local_main_process_first():
        ...     # This will be printed first by local process 0 then in a seemingly
        ...     # random order by the other processes.
        ...     print(f"This will be printed by process {state.local_process_index}")
        ```
        """
        yield from self._goes_first(self.is_local_main_process)

    def on_main_process(self, function: Callable[..., Any]=None):
        """
        Decorator that only runs the decorated function on the main process.

        Args:
            function (`Callable`): The function to decorate.

        Example:

        ```python
        >>> from accelerate.state import PartialState

        >>> state = PartialState()


        >>> @state.on_main_process
        ... def print_something():
        ...     print("This will be printed by process 0 only.")


        >>> print_something()
        "This will be printed by process 0 only"
        ```
        """
        if not self.initialized:
            raise ValueError('The `PartialState` or `Accelerator` must be initialized before calling this function.')
        if self.is_main_process or not self.use_distributed:
            return function
        return do_nothing

    def on_local_main_process(self, function: Callable[..., Any]=None):
        """
        Decorator that only runs the decorated function on the local main process.

        Args:
            function (`Callable`): The function to decorate.

        Example:
        ```python
        # Assume we have 2 servers with 4 processes each.
        from accelerate.state import PartialState

        state = PartialState()


        @state.on_local_main_process
        def print_something():
            print("This will be printed by process 0 only on each server.")


        print_something()
        # On server 1:
        "This will be printed by process 0 only"
        # On server 2:
        "This will be printed by process 0 only"
        ```
        """
        if self.is_local_main_process or not self.use_distributed:
            return function
        return do_nothing

    def on_last_process(self, function: Callable[..., Any]):
        """
        Decorator that only runs the decorated function on the last process.

        Args:
            function (`Callable`): The function to decorate.

        Example:
        ```python
        # Assume we have 4 processes.
        from accelerate.state import PartialState

        state = PartialState()


        @state.on_last_process
        def print_something():
            print(f"Printed on process {state.process_index}")


        print_something()
        "Printed on process 3"
        ```
        """
        if self.is_last_process or not self.use_distributed:
            return function
        return do_nothing

    def on_process(self, function: Callable[..., Any]=None, process_index: int=None):
        """
        Decorator that only runs the decorated function on the process with the given index.

        Args:
            function (`Callable`, `optional`):
                The function to decorate.
            process_index (`int`, `optional`):
                The index of the process on which to run the function.

        Example:
        ```python
        # Assume we have 4 processes.
        from accelerate.state import PartialState

        state = PartialState()


        @state.on_process(process_index=2)
        def print_something():
            print(f"Printed on process {state.process_index}")


        print_something()
        "Printed on process 2"
        ```
        """
        if function is None:
            return partial(self.on_process, process_index=process_index)
        if self.process_index == process_index or not self.use_distributed:
            return function
        return do_nothing

    def on_local_process(self, function: Callable[..., Any]=None, local_process_index: int=None):
        """
        Decorator that only runs the decorated function on the process with the given index on the current node.

        Args:
            function (`Callable`, *optional*):
                The function to decorate.
            local_process_index (`int`, *optional*):
                The index of the local process on which to run the function.

        Example:
        ```python
        # Assume we have 2 servers with 4 processes each.
        from accelerate import Accelerator

        accelerator = Accelerator()


        @accelerator.on_local_process(local_process_index=2)
        def print_something():
            print(f"Printed on process {accelerator.local_process_index}")


        print_something()
        # On server 1:
        "Printed on process 2"
        # On server 2:
        "Printed on process 2"
        ```
        """
        if function is None:
            return partial(self.on_local_process, local_process_index=local_process_index)
        if self.local_process_index == local_process_index or not self.use_distributed:
            return function
        return do_nothing

    def print(self, *args, **kwargs):
        if self.is_local_main_process:
            print(*args, **kwargs)

    @property
    def default_device(self) -> torch.device:
        """
        Returns the default device which is:
        - MPS if `torch.backends.mps.is_available()` and `torch.backends.mps.is_built()` both return True.
        - CUDA if `torch.cuda.is_available()`
        - NPU if `is_npu_available()`
        - CPU otherwise
        """
        if is_mps_available():
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        elif is_xpu_available():
            return torch.device('xpu:0')
        elif is_npu_available():
            return torch.device('npu')
        else:
            return torch.device('cpu')