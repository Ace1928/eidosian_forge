from typing import Optional
class AutoscalerPrometheusMetrics(object):

    def __init__(self, session_name: str=None):
        pass

    def __getattr__(self, attr):
        return NullMetric()