import os
from threading import Lock
import time
import types
from typing import (
import warnings
from . import values  # retain this import style for testability
from .context_managers import ExceptionCounter, InprogressTracker, Timer
from .metrics_core import (
from .registry import Collector, CollectorRegistry, REGISTRY
from .samples import Exemplar, Sample
from .utils import floatToGoString, INF
class MetricWrapperBase(Collector):
    _type: Optional[str] = None
    _reserved_labelnames: Sequence[str] = ()

    def _is_observable(self):
        return not self._labelnames or (self._labelnames and self._labelvalues)

    def _raise_if_not_observable(self):
        if not self._is_observable():
            raise ValueError('%s metric is missing label values' % str(self._type))

    def _is_parent(self):
        return self._labelnames and (not self._labelvalues)

    def _get_metric(self):
        return Metric(self._name, self._documentation, self._type, self._unit)

    def describe(self) -> Iterable[Metric]:
        return [self._get_metric()]

    def collect(self) -> Iterable[Metric]:
        metric = self._get_metric()
        for suffix, labels, value, timestamp, exemplar in self._samples():
            metric.add_sample(self._name + suffix, labels, value, timestamp, exemplar)
        return [metric]

    def __str__(self) -> str:
        return f'{self._type}:{self._name}'

    def __repr__(self) -> str:
        metric_type = type(self)
        return f'{metric_type.__module__}.{metric_type.__name__}({self._name})'

    def __init__(self: T, name: str, documentation: str, labelnames: Iterable[str]=(), namespace: str='', subsystem: str='', unit: str='', registry: Optional[CollectorRegistry]=REGISTRY, _labelvalues: Optional[Sequence[str]]=None) -> None:
        self._name = _build_full_name(self._type, name, namespace, subsystem, unit)
        self._labelnames = _validate_labelnames(self, labelnames)
        self._labelvalues = tuple(_labelvalues or ())
        self._kwargs: Dict[str, Any] = {}
        self._documentation = documentation
        self._unit = unit
        if not METRIC_NAME_RE.match(self._name):
            raise ValueError('Invalid metric name: ' + self._name)
        if self._is_parent():
            self._lock = Lock()
            self._metrics: Dict[Sequence[str], T] = {}
        if self._is_observable():
            self._metric_init()
        if not self._labelvalues:
            if registry:
                registry.register(self)

    def labels(self: T, *labelvalues: Any, **labelkwargs: Any) -> T:
        """Return the child for the given labelset.

        All metrics can have labels, allowing grouping of related time series.
        Taking a counter as an example:

            from prometheus_client import Counter

            c = Counter('my_requests_total', 'HTTP Failures', ['method', 'endpoint'])
            c.labels('get', '/').inc()
            c.labels('post', '/submit').inc()

        Labels can also be provided as keyword arguments:

            from prometheus_client import Counter

            c = Counter('my_requests_total', 'HTTP Failures', ['method', 'endpoint'])
            c.labels(method='get', endpoint='/').inc()
            c.labels(method='post', endpoint='/submit').inc()

        See the best practices on [naming](http://prometheus.io/docs/practices/naming/)
        and [labels](http://prometheus.io/docs/practices/instrumentation/#use-labels).
        """
        if not self._labelnames:
            raise ValueError('No label names were set when constructing %s' % self)
        if self._labelvalues:
            raise ValueError('{} already has labels set ({}); can not chain calls to .labels()'.format(self, dict(zip(self._labelnames, self._labelvalues))))
        if labelvalues and labelkwargs:
            raise ValueError("Can't pass both *args and **kwargs")
        if labelkwargs:
            if sorted(labelkwargs) != sorted(self._labelnames):
                raise ValueError('Incorrect label names')
            labelvalues = tuple((str(labelkwargs[l]) for l in self._labelnames))
        else:
            if len(labelvalues) != len(self._labelnames):
                raise ValueError('Incorrect label count')
            labelvalues = tuple((str(l) for l in labelvalues))
        with self._lock:
            if labelvalues not in self._metrics:
                self._metrics[labelvalues] = self.__class__(self._name, documentation=self._documentation, labelnames=self._labelnames, unit=self._unit, _labelvalues=labelvalues, **self._kwargs)
            return self._metrics[labelvalues]

    def remove(self, *labelvalues: Any) -> None:
        if 'prometheus_multiproc_dir' in os.environ or 'PROMETHEUS_MULTIPROC_DIR' in os.environ:
            warnings.warn('Removal of labels has not been implemented in  multi-process mode yet.', UserWarning)
        if not self._labelnames:
            raise ValueError('No label names were set when constructing %s' % self)
        'Remove the given labelset from the metric.'
        if len(labelvalues) != len(self._labelnames):
            raise ValueError('Incorrect label count (expected %d, got %s)' % (len(self._labelnames), labelvalues))
        labelvalues = tuple((str(l) for l in labelvalues))
        with self._lock:
            del self._metrics[labelvalues]

    def clear(self) -> None:
        """Remove all labelsets from the metric"""
        if 'prometheus_multiproc_dir' in os.environ or 'PROMETHEUS_MULTIPROC_DIR' in os.environ:
            warnings.warn('Clearing labels has not been implemented in multi-process mode yet', UserWarning)
        with self._lock:
            self._metrics = {}

    def _samples(self) -> Iterable[Sample]:
        if self._is_parent():
            return self._multi_samples()
        else:
            return self._child_samples()

    def _multi_samples(self) -> Iterable[Sample]:
        with self._lock:
            metrics = self._metrics.copy()
        for labels, metric in metrics.items():
            series_labels = list(zip(self._labelnames, labels))
            for suffix, sample_labels, value, timestamp, exemplar in metric._samples():
                yield Sample(suffix, dict(series_labels + list(sample_labels.items())), value, timestamp, exemplar)

    def _child_samples(self) -> Iterable[Sample]:
        raise NotImplementedError('_child_samples() must be implemented by %r' % self)

    def _metric_init(self):
        """
        Initialize the metric object as a child, i.e. when it has labels (if any) set.

        This is factored as a separate function to allow for deferred initialization.
        """
        raise NotImplementedError('_metric_init() must be implemented by %r' % self)