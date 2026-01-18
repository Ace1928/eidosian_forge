import abc
from .stat import AbstractStat
class AbstractCompoundStat(AbstractStat):
    """
    A compound stat is a stat where a single measurement and associated
    data structure feeds many metrics. This is the example for a
    histogram which has many associated percentiles.
    """
    __metaclass__ = abc.ABCMeta

    def stats(self):
        """
        Return list of NamedMeasurable
        """
        raise NotImplementedError