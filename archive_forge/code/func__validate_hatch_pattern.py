import numpy as np
from matplotlib import _api
from matplotlib.path import Path
def _validate_hatch_pattern(hatch):
    valid_hatch_patterns = set('-+|/\\xXoO.*')
    if hatch is not None:
        invalids = set(hatch).difference(valid_hatch_patterns)
        if invalids:
            valid = ''.join(sorted(valid_hatch_patterns))
            invalids = ''.join(sorted(invalids))
            _api.warn_deprecated('3.4', removal='3.9', message=f'hatch must consist of a string of "{valid}" or None, but found the following invalid values "{invalids}". Passing invalid values is deprecated since %(since)s and will become an error %(removal)s.')