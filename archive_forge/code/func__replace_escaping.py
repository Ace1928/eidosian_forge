import io as StringIO
import math
import re
from ..metrics_core import Metric, METRIC_LABEL_NAME_RE
from ..samples import Exemplar, Sample, Timestamp
from ..utils import floatToGoString
def _replace_escaping(s):
    return ESCAPING_RE.sub(_replace_escape_sequence, s)