from collections.abc import MutableMapping
from collections import ChainMap as _ChainMap
import functools
import io
import itertools
import os
import re
import sys
import warnings
Get an option value.

        Unless `fallback` is provided, `None` will be returned if the option
        is not found.

        