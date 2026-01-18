from aiokafka.metrics.measurable import AnonMeasurable
from aiokafka.metrics.compound_stat import AbstractCompoundStat, NamedMeasurable
from .histogram import Histogram
from .sampled_stat import AbstractSampledStat
class HistogramSample(AbstractSampledStat.Sample):

    def __init__(self, scheme, now):
        super(Percentiles.HistogramSample, self).__init__(0.0, now)
        self.histogram = Histogram(scheme)