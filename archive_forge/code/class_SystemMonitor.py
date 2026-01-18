import datetime
import logging
import queue
import threading
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Deque, Dict, List, Optional, Tuple
from .assets.asset_registry import asset_registry
from .assets.interfaces import Asset, Interface
from .assets.open_metrics import OpenMetrics
from .system_info import SystemInfo
class SystemMonitor:
    PUBLISHING_INTERVAL_DELAY_FACTOR = 2

    def __init__(self, settings: 'SettingsStatic', interface: 'Interface') -> None:
        self._shutdown_event: threading.Event = threading.Event()
        self._process: Optional[threading.Thread] = None
        self.settings = settings
        sampling_interval: float = float(max(0.1, self.settings._stats_sample_rate_seconds))
        samples_to_aggregate: int = min(30, max(1, self.settings._stats_samples_to_average))
        self.publishing_interval: float = sampling_interval * samples_to_aggregate
        self.join_assets: bool = self.settings._stats_join_assets
        self.backend_interface = interface
        self.asset_interface: Optional[AssetInterface] = AssetInterface() if self.join_assets else None
        self.assets: List[Asset] = self._get_assets()
        self.assets.extend(self._get_open_metrics_assets())
        self.system_info: SystemInfo = SystemInfo(settings=self.settings, interface=interface)
        self.buffer: Dict[str, Deque[Tuple[float, float]]] = defaultdict(lambda: deque([], maxlen=self.settings._stats_buffer_size))

    def _get_assets(self) -> List['Asset']:
        return [asset_class(interface=self.asset_interface or self.backend_interface, settings=self.settings, shutdown_event=self._shutdown_event) for asset_class in asset_registry]

    def _get_open_metrics_assets(self) -> List['Asset']:
        open_metrics_endpoints = self.settings._stats_open_metrics_endpoints
        if not open_metrics_endpoints:
            return []
        assets: List[Asset] = []
        for name, endpoint in open_metrics_endpoints.items():
            if not OpenMetrics.is_available(url=endpoint):
                continue
            logger.debug(f'Monitoring OpenMetrics endpoint: {endpoint}')
            open_metrics = OpenMetrics(interface=self.asset_interface or self.backend_interface, settings=self.settings, shutdown_event=self._shutdown_event, name=name, url=endpoint)
            assets.append(open_metrics)
        return assets

    def aggregate_and_publish_asset_metrics(self) -> None:
        if self.asset_interface is None:
            return None
        size = self.asset_interface.metrics_queue.qsize()
        aggregated_metrics = {}
        for _ in range(size):
            item = self.asset_interface.metrics_queue.get()
            aggregated_metrics.update(item)
        if aggregated_metrics:
            t = datetime.datetime.now().timestamp()
            for k, v in aggregated_metrics.items():
                self.buffer[k].append((t, v))
            self.backend_interface.publish_stats(aggregated_metrics)

    def publish_telemetry(self) -> None:
        if self.asset_interface is None:
            return None
        while not self.asset_interface.telemetry_queue.empty():
            telemetry_record = self.asset_interface.telemetry_queue.get()
            self.backend_interface._publish_telemetry(telemetry_record)

    def _start(self) -> None:
        logger.info('Starting system asset monitoring threads')
        for asset in self.assets:
            asset.start()
        if not (self.join_assets and self.asset_interface is not None):
            return None
        self._shutdown_event.wait(self.publishing_interval * self.PUBLISHING_INTERVAL_DELAY_FACTOR)
        logger.debug('Starting system metrics aggregation loop')
        while not self._shutdown_event.is_set():
            self.publish_telemetry()
            self.aggregate_and_publish_asset_metrics()
            self._shutdown_event.wait(self.publishing_interval)
        logger.debug('Finished system metrics aggregation loop')
        try:
            logger.debug('Publishing last batch of metrics')
            self.publish_telemetry()
            self.aggregate_and_publish_asset_metrics()
        except Exception as e:
            logger.error(f'Error publishing last batch of metrics: {e}')

    def start(self) -> None:
        self._shutdown_event.clear()
        if self._process is not None:
            return None
        logger.info('Starting system monitor')
        self._process = threading.Thread(target=self._start, daemon=True, name='SystemMonitor')
        self._process.start()

    def finish(self) -> None:
        if self._process is None:
            return None
        logger.info('Stopping system monitor')
        self._shutdown_event.set()
        for asset in self.assets:
            asset.finish()
        try:
            self._process.join()
        except Exception as e:
            logger.error(f'Error joining system monitor process: {e}')
        self._process = None

    def probe(self, publish: bool=True) -> None:
        logger.info('Collecting system info')
        hardware_info: dict = {k: v for d in [asset.probe() for asset in self.assets] for k, v in d.items()}
        software_info: dict = self.system_info.probe()
        system_info = {**software_info, **hardware_info}
        logger.debug(system_info)
        logger.info('Finished collecting system info')
        if publish:
            logger.info('Publishing system info')
            self.system_info.publish(system_info)
            logger.info('Finished publishing system info')