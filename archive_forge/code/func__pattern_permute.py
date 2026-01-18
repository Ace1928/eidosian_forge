from __future__ import annotations
import re
from . import Image, _imagingmorph
def _pattern_permute(self, basic_pattern, options, basic_result):
    """pattern_permute takes a basic pattern and its result and clones
        the pattern according to the modifications described in the $options
        parameter. It returns a list of all cloned patterns."""
    patterns = [(basic_pattern, basic_result)]
    if '4' in options:
        res = patterns[-1][1]
        for i in range(4):
            patterns.append((self._string_permute(patterns[-1][0], ROTATION_MATRIX), res))
    if 'M' in options:
        n = len(patterns)
        for pattern, res in patterns[:n]:
            patterns.append((self._string_permute(pattern, MIRROR_MATRIX), res))
    if 'N' in options:
        n = len(patterns)
        for pattern, res in patterns[:n]:
            pattern = pattern.replace('0', 'Z').replace('1', '0').replace('Z', '1')
            res = 1 - int(res)
            patterns.append((pattern, res))
    return patterns