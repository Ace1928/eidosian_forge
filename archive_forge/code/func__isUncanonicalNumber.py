import io as StringIO
import math
import re
from ..metrics_core import Metric, METRIC_LABEL_NAME_RE
from ..samples import Exemplar, Sample, Timestamp
from ..utils import floatToGoString
def _isUncanonicalNumber(s):
    f = float(s)
    if f not in _CANONICAL_NUMBERS:
        return False
    return s != floatToGoString(f)