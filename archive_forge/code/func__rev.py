import decimal
import json as _json
import sys
import re
from functools import reduce
from _plotly_utils.optional_imports import get_module
from _plotly_utils.basevalidators import ImageUriValidator
def _rev(l):
    return [s[::-1] for s in l][::-1]