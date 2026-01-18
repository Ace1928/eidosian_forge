import io as StringIO
import math
import re
from ..metrics_core import Metric, METRIC_LABEL_NAME_RE
from ..samples import Exemplar, Sample, Timestamp
from ..utils import floatToGoString
def _check_histogram(samples, name):
    group = None
    timestamp = None

    def do_checks():
        if bucket != float('+Inf'):
            raise ValueError('+Inf bucket missing: ' + name)
        if count is not None and value != count:
            raise ValueError('Count does not match +Inf value: ' + name)
        if has_sum and count is None:
            raise ValueError('_count must be present if _sum is present: ' + name)
        if has_gsum and count is None:
            raise ValueError('_gcount must be present if _gsum is present: ' + name)
        if not (has_sum or has_gsum) and count is not None:
            raise ValueError('_sum/_gsum must be present if _count is present: ' + name)
        if has_negative_buckets and has_sum:
            raise ValueError('Cannot have _sum with negative buckets: ' + name)
        if not has_negative_buckets and has_negative_gsum:
            raise ValueError('Cannot have negative _gsum with non-negative buckets: ' + name)
    for s in samples:
        suffix = s.name[len(name):]
        g = _group_for_sample(s, name, 'histogram')
        if g != group or s.timestamp != timestamp:
            if group is not None:
                do_checks()
            count = None
            bucket = None
            has_negative_buckets = False
            has_sum = False
            has_gsum = False
            has_negative_gsum = False
            value = 0
        group = g
        timestamp = s.timestamp
        if suffix == '_bucket':
            b = float(s.labels['le'])
            if b < 0:
                has_negative_buckets = True
            if bucket is not None and b <= bucket:
                raise ValueError('Buckets out of order: ' + name)
            if s.value < value:
                raise ValueError('Bucket values out of order: ' + name)
            bucket = b
            value = s.value
        elif suffix in ['_count', '_gcount']:
            count = s.value
        elif suffix in ['_sum']:
            has_sum = True
        elif suffix in ['_gsum']:
            has_gsum = True
            if s.value < 0:
                has_negative_gsum = True
    if group is not None:
        do_checks()