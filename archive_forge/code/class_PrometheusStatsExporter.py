import re
from prometheus_client import start_http_server
from prometheus_client.core import (
from opencensus.common.transports import sync
from opencensus.stats import aggregation_data as aggregation_data_module
from opencensus.stats import base_exporter
import logging
class PrometheusStatsExporter(base_exporter.StatsExporter):
    """Exporter exports stats to Prometheus, users need
        to register the exporter as an HTTP Handler to be
        able to export.
    :type options:
        :class:`~opencensus.ext.prometheus.stats_exporter.Options`
    :param options: An options object with the parameters to instantiate the
                         prometheus exporter.
    :type gatherer: :class:`~prometheus_client.core.CollectorRegistry`
    :param gatherer: A Prometheus collector registry instance.
    :type transport:
        :class:`opencensus.common.transports.sync.SyncTransport` or
        :class:`opencensus.common.transports.async_.AsyncTransport`
    :param transport: An instance of a Transpor to send data with.
    :type collector:
        :class:`~opencensus.ext.prometheus.stats_exporter.Collector`
    :param collector: An instance of the Prometheus Collector object.
    """

    def __init__(self, options, gatherer, transport=sync.SyncTransport, collector=Collector()):
        self._options = options
        self._gatherer = gatherer
        self._collector = collector
        self._transport = transport(self)
        self.serve_http()
        REGISTRY.register(self._collector)

    @property
    def transport(self):
        """The transport way to be sent data to server
        (default is sync).
        """
        return self._transport

    @property
    def collector(self):
        """Collector class instance to be used
        to communicate with Prometheus
        """
        return self._collector

    @property
    def gatherer(self):
        """Prometheus Collector Registry instance"""
        return self._gatherer

    @property
    def options(self):
        """Options to be used to configure the exporter"""
        return self._options

    def export(self, view_data):
        """export send the data to the transport class
        in order to be sent to Prometheus in a sync or async way.
        """
        if view_data is not None:
            self.transport.export(view_data)

    def on_register_view(self, view):
        return NotImplementedError('Not supported by Prometheus')

    def emit(self, view_data):
        """Emit exports to the Prometheus if view data has one or more rows.
        Each OpenCensus AggregationData will be converted to
        corresponding Prometheus Metric: SumData will be converted
        to Untyped Metric, CountData will be a Counter Metric
        DistributionData will be a Histogram Metric.
        """
        for v_data in view_data:
            if v_data.tag_value_aggregation_data_map is None:
                v_data.tag_value_aggregation_data_map = {}
            self.collector.add_view_data(v_data)

    def serve_http(self):
        """serve_http serves the Prometheus endpoint."""
        address = str(self.options.address)
        kwargs = {'addr': address} if address else {}
        start_http_server(port=self.options.port, **kwargs)