import threading
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union
import wandb
from .aggregators import aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
class IPUStats:
    """Stats for Graphcore IPU devices."""
    name = 'ipu.{}.{}'
    samples: 'Deque[dict]'
    variable_metric_keys = {'average board temp', 'average die temp', 'clock', 'ipu power', 'ipu utilisation', 'ipu utilisation (session)'}

    def __init__(self, pid: int, gc_ipu_info: Optional[Any]=None) -> None:
        self.samples: Deque[dict] = deque()
        if gc_ipu_info is None:
            if not gcipuinfo:
                raise ImportError('Monitoring IPU stats requires gcipuinfo to be installed')
            self._gc_ipu_info = gcipuinfo.gcipuinfo()
        else:
            self._gc_ipu_info = gc_ipu_info
        self._gc_ipu_info.setUpdateMode(True)
        self._pid = pid
        self._devices_called: Set[str] = set()

    @staticmethod
    def parse_metric(key: str, value: str) -> Optional[Tuple[str, Union[int, float]]]:
        metric_suffixes = {'temp': 'C', 'clock': 'MHz', 'power': 'W', 'utilisation': '%', 'utilisation (session)': '%', 'speed': 'GT/s'}
        for metric, suffix in metric_suffixes.items():
            if key.endswith(metric) and value.endswith(suffix):
                value = value[:-len(suffix)]
                key = f'{key} ({suffix})'
        try:
            float_value = float(value)
            num_value = int(float_value) if float_value.is_integer() else float_value
        except ValueError:
            return None
        return (key, num_value)

    def sample(self) -> None:
        try:
            stats = {}
            devices = self._gc_ipu_info.getDevices()
            for device in devices:
                device_metrics: Dict[str, str] = dict(device)
                pid = device_metrics.get('user process id')
                if pid is None or int(pid) != self._pid:
                    continue
                device_id = device_metrics.get('id')
                initial_call = device_id not in self._devices_called
                if device_id is not None:
                    self._devices_called.add(device_id)
                for key, value in device_metrics.items():
                    log_metric = initial_call or key in self.variable_metric_keys
                    if not log_metric:
                        continue
                    parsed = self.parse_metric(key, value)
                    if parsed is None:
                        continue
                    parsed_key, parsed_value = parsed
                    stats[self.name.format(device_id, parsed_key)] = parsed_value
            self.samples.append(stats)
        except Exception as e:
            wandb.termwarn(f'IPU stats error {e}', repeat=False)

    def clear(self) -> None:
        self.samples.clear()

    def aggregate(self) -> dict:
        if not self.samples:
            return {}
        stats = {}
        for key in self.samples[0].keys():
            samples = [s[key] for s in self.samples if key in s]
            aggregate = aggregate_mean(samples)
            stats[key] = aggregate
        return stats