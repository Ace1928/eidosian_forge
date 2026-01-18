import json
import logging
import shutil
import subprocess
import sys
import threading
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List, Union
from wandb.sdk.lib import telemetry
from .aggregators import aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
@asset_registry.register
class GPUAMD:
    """GPUAMD is a class for monitoring AMD GPU devices.

    Uses AMD's rocm_smi tool to get GPU stats.
    For the list of supported environments and devices, see
    https://github.com/RadeonOpenCompute/ROCm/blob/develop/docs/deploy/
    """

    def __init__(self, interface: 'Interface', settings: 'SettingsStatic', shutdown_event: threading.Event) -> None:
        self.name = self.__class__.__name__.lower()
        self.metrics: List[Metric] = [GPUAMDStats()]
        self.metrics_monitor = MetricsMonitor(self.name, self.metrics, interface, settings, shutdown_event)
        telemetry_record = telemetry.TelemetryRecord()
        telemetry_record.env.amd_gpu = True
        interface._publish_telemetry(telemetry_record)

    @classmethod
    def is_available(cls) -> bool:
        rocm_smi_available = shutil.which(ROCM_SMI_CMD) is not None
        if not rocm_smi_available:
            return False
        is_driver_initialized = False
        try:
            with open('/sys/module/amdgpu/initstate') as file:
                file_content = file.read()
                if 'live' in file_content:
                    is_driver_initialized = True
        except FileNotFoundError:
            pass
        can_read_rocm_smi = False
        try:
            if get_rocm_smi_stats():
                can_read_rocm_smi = True
        except Exception:
            pass
        return is_driver_initialized and can_read_rocm_smi

    def start(self) -> None:
        self.metrics_monitor.start()

    def finish(self) -> None:
        self.metrics_monitor.finish()

    def probe(self) -> dict:
        info: _InfoDict = {}
        try:
            stats = get_rocm_smi_stats()
            info['gpu_count'] = len([key for key in stats.keys() if key.startswith('card')])
            key_mapping = {'id': 'GPU ID', 'unique_id': 'Unique ID', 'vbios_version': 'VBIOS version', 'performance_level': 'Performance Level', 'gpu_overdrive': 'GPU OverDrive value (%)', 'gpu_memory_overdrive': 'GPU Memory OverDrive value (%)', 'max_power': 'Max Graphics Package Power (W)', 'series': 'Card series', 'model': 'Card model', 'vendor': 'Card vendor', 'sku': 'Card SKU', 'sclk_range': 'Valid sclk range', 'mclk_range': 'Valid mclk range'}
            info['gpu_devices'] = [{k: stats[key][v] for k, v in key_mapping.items() if stats[key].get(v)} for key in stats.keys() if key.startswith('card')]
        except Exception as e:
            logger.exception(f'GPUAMD probe error: {e}')
        return info