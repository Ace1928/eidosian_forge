import collections
import dataclasses
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from wandb.sdk.lib import telemetry
from .aggregators import aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
@asset_registry.register
class Trainium:

    def __init__(self, interface: 'Interface', settings: 'SettingsStatic', shutdown_event: threading.Event) -> None:
        self.name = self.__class__.__name__.lower()
        self.metrics: List[Metric] = [NeuronCoreStats(settings._stats_pid, settings._stats_neuron_monitor_config_path)]
        self.metrics_monitor = MetricsMonitor(self.name, self.metrics, interface, settings, shutdown_event)
        telemetry_record = telemetry.TelemetryRecord()
        telemetry_record.env.trainium = True
        interface._publish_telemetry(telemetry_record)

    @classmethod
    def is_available(cls) -> bool:
        if not pathlib.Path(NEURON_LS_COMMAND[0]).exists():
            return False
        try:
            output = subprocess.check_output(NEURON_LS_COMMAND, universal_newlines=True, stderr=subprocess.DEVNULL).strip()
            if len(json.loads(output)) > 0:
                return True
        except (OSError, ValueError, TypeError, subprocess.CalledProcessError):
            pass
        return False

    def start(self) -> None:
        self.metrics_monitor.start()

    def finish(self) -> None:
        self.metrics_monitor.finish()

    def probe(self) -> dict:
        try:
            self.metrics[0].check_neuron_monitor_config()
            neuron_hardware_info: dict = {}
            command = [NEURON_MONITOR_PATH, '-c', self.metrics[0].neuron_monitor_config_path]
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None) as process:
                while True:
                    if process.stdout is None:
                        time.sleep(0.1)
                        continue
                    raw_data = process.stdout.readline()
                    if raw_data:
                        parsed_data = json.loads(raw_data)
                        neuron_hardware_info = parsed_data.get('neuron_hardware_info', {})
                        neuron_hardware_info.pop('error', None)
                        break
            try:
                process.kill()
                process.wait()
            except:
                pass
            return {self.name: neuron_hardware_info}
        except Exception as e:
            logger.error('neuron-monitor failed: %s' % e)
            return {}