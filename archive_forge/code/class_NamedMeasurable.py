import abc
from .stat import AbstractStat
class NamedMeasurable(object):

    def __init__(self, metric_name, measurable_stat):
        self._name = metric_name
        self._stat = measurable_stat

    @property
    def name(self):
        return self._name

    @property
    def stat(self):
        return self._stat