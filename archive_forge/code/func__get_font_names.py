from __future__ import annotations
from collections.abc import (
import functools
import itertools
import re
from typing import (
import warnings
import numpy as np
from pandas._libs.lib import is_list_like
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes import missing
from pandas.core.dtypes.common import (
from pandas import (
import pandas.core.common as com
from pandas.core.shared_docs import _shared_docs
from pandas.io.formats._color_data import CSS4_COLORS
from pandas.io.formats.css import (
from pandas.io.formats.format import get_level_lengths
from pandas.io.formats.printing import pprint_thing
def _get_font_names(self, props: Mapping[str, str]) -> Sequence[str]:
    font_names_tmp = re.findall('(?x)\n            (\n            "(?:[^"]|\\\\")+"\n            |\n            \'(?:[^\']|\\\\\')+\'\n            |\n            [^\'",]+\n            )(?=,|\\s*$)\n        ', props.get('font-family', ''))
    font_names = []
    for name in font_names_tmp:
        if name[:1] == '"':
            name = name[1:-1].replace('\\"', '"')
        elif name[:1] == "'":
            name = name[1:-1].replace("\\'", "'")
        else:
            name = name.strip()
        if name:
            font_names.append(name)
    return font_names