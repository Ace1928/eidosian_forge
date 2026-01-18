from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, List, Optional, TypeVar, Union
import torch
from typing_extensions import override
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_1
from lightning_fabric.utilities.rank_zero import rank_zero_only, rank_zero_warn
class Throughput:
    """Computes throughput.

    +------------------------+-------------------------------------------------------------------------------------+
    | Key                    | Value                                                                               |
    +========================+=====================================================================================+
    | batches_per_sec        | Rolling average (over ``window_size`` most recent updates) of the number of batches |
    |                        | processed per second                                                                |
    +--------------------------+-----------------------------------------------------------------------------------+
    | samples_per_sec        | Rolling average (over ``window_size`` most recent updates) of the number of samples |
    |                        | processed per second                                                                |
    +--------------------------+-----------------------------------------------------------------------------------+
    | items_per_sec          | Rolling average (over ``window_size`` most recent updates) of the number of items   |
    |                        | processed per second                                                                |
    +--------------------------+-----------------------------------------------------------------------------------+
    | flpps_per_sec          | Rolling average (over ``window_size`` most recent updates) of the number of flops   |
    |                        | processed per second                                                                |
    +--------------------------+-----------------------------------------------------------------------------------+
    | device/batches_per_sec | batches_per_sec divided by world size                                               |
    +--------------------------+-----------------------------------------------------------------------------------+
    | device/samples_per_sec | samples_per_sec divided by world size                                               |
    +--------------------------+-----------------------------------------------------------------------------------+
    | device/items_per_sec   | items_per_sec divided by world size. This may include padding depending on the data |
    +--------------------------+-----------------------------------------------------------------------------------+
    | device/flops_per_sec   | flops_per_sec divided by world size.                                                |
    +--------------------------+-----------------------------------------------------------------------------------+
    | device/mfu             | device/flops_per_sec divided by world size.                                         |
    +--------------------------+-----------------------------------------------------------------------------------+
    | time                   | Total elapsed time                                                                  |
    +--------------------------+-----------------------------------------------------------------------------------+
    | batches                | Total batches seen                                                                  |
    +--------------------------+-----------------------------------------------------------------------------------+
    | samples                | Total samples seen                                                                  |
    +--------------------------+-----------------------------------------------------------------------------------+
    | lengths                | Total items seen                                                                    |
    +--------------------------+-----------------------------------------------------------------------------------+

    Example::

        throughput = Throughput()
        t0 = time()
        for i in range(1000):
            do_work()
            if torch.cuda.is_available(): torch.cuda.synchronize()  # required or else time() won't be correct
            throughput.update(time=time() - t0, samples=i)
            if i % 10 == 0:
                print(throughput.compute())

    Notes:
        - The implementation assumes that devices FLOPs are all the same as it normalizes by the world size and only
          takes a single ``available_flops`` value.
        - items_per_sec, flops_per_sec and MFU do not account for padding if present. We suggest using
          samples_per_sec or batches_per_sec to measure throughput under this circumstance.

    Args:
        available_flops: Number of theoretical flops available for a single device.
        world_size: Number of devices available across hosts. Global metrics are not included if the world size is 1.
        window_size: Number of batches to use for a rolling average.
        separator: Key separator to use when creating per-device and global metrics.

    """

    def __init__(self, available_flops: Optional[float]=None, world_size: int=1, window_size: int=100, separator: str='/') -> None:
        self.available_flops = available_flops
        self.separator = separator
        assert world_size > 0
        self.world_size = world_size
        assert window_size > 1
        self._time: _MonotonicWindow[float] = _MonotonicWindow(maxlen=window_size)
        self._batches: _MonotonicWindow[int] = _MonotonicWindow(maxlen=window_size)
        self._samples: _MonotonicWindow[int] = _MonotonicWindow(maxlen=window_size)
        self._lengths: _MonotonicWindow[int] = _MonotonicWindow(maxlen=window_size)
        self._flops: Deque[int] = deque(maxlen=window_size)

    def update(self, *, time: float, batches: int, samples: int, lengths: Optional[int]=None, flops: Optional[int]=None) -> None:
        """Update throughput metrics.

        Args:
            time: Total elapsed time in seconds. It should monotonically increase by the iteration time with each
                call.
            batches: Total batches seen per device. It should monotonically increase with each call.
            samples: Total samples seen per device. It should monotonically increase by the batch size with each call.
            lengths: Total length of the samples seen. It should monotonically increase by the lengths of a batch with
                each call.
            flops: Flops elapased per device since last ``update()`` call. You can easily compute this by using
                :func:`measure_flops` and multiplying it by the number of batches that have been processed.
                The value might be different in each device if the batch size is not the same.

        """
        self._time.append(time)
        if samples < batches:
            raise ValueError(f'Expected samples ({samples}) to be greater or equal than batches ({batches})')
        self._batches.append(batches)
        self._samples.append(samples)
        if lengths is not None:
            if lengths < samples:
                raise ValueError(f'Expected lengths ({lengths}) to be greater or equal than samples ({samples})')
            self._lengths.append(lengths)
            if len(self._samples) != len(self._lengths):
                raise RuntimeError(f'If lengths are passed ({len(self._lengths)}), there needs to be the same number of samples ({len(self._samples)})')
        if flops is not None:
            self._flops.append(flops * self.world_size)

    def compute(self) -> _THROUGHPUT_METRICS:
        """Compute throughput metrics."""
        metrics = {'time': self._time[-1], 'batches': self._batches[-1], 'samples': self._samples[-1]}
        if self._lengths:
            metrics['lengths'] = self._lengths[-1]
        add_global_metrics = self.world_size > 1
        if len(self._time) == self._time.maxlen:
            elapsed_time = self._time[-1] - self._time[0]
            elapsed_batches = self._batches[-1] - self._batches[0]
            elapsed_samples = self._samples[-1] - self._samples[0]
            dev_samples_per_sec = elapsed_samples / elapsed_time
            dev_batches_per_sec = elapsed_batches / elapsed_time
            metrics.update({f'device{self.separator}batches_per_sec': elapsed_batches / elapsed_time, f'device{self.separator}samples_per_sec': dev_samples_per_sec})
            if add_global_metrics:
                samples_per_sec = dev_batches_per_sec * self.world_size
                metrics.update({'batches_per_sec': samples_per_sec, 'samples_per_sec': dev_samples_per_sec * self.world_size})
            if len(self._lengths) == self._lengths.maxlen:
                elapsed_lengths = self._lengths[-1] - self._lengths[0]
                dev_items_per_sec = elapsed_lengths / elapsed_time
                metrics[f'device{self.separator}items_per_sec'] = dev_items_per_sec
                if add_global_metrics:
                    items_per_sec = dev_items_per_sec * self.world_size
                    metrics['items_per_sec'] = items_per_sec
        if len(self._flops) == self._flops.maxlen:
            elapsed_flops = sum(self._flops) - self._flops[0]
            elapsed_time = self._time[-1] - self._time[0]
            flops_per_sec = elapsed_flops / elapsed_time
            dev_flops_per_sec = flops_per_sec / self.world_size
            if add_global_metrics:
                metrics['flops_per_sec'] = flops_per_sec
            metrics[f'device{self.separator}flops_per_sec'] = dev_flops_per_sec
            if self.available_flops:
                metrics[f'device{self.separator}mfu'] = dev_flops_per_sec / self.available_flops
        return metrics

    def reset(self) -> None:
        self._time.clear()
        self._batches.clear()
        self._samples.clear()
        self._lengths.clear()
        self._flops.clear()