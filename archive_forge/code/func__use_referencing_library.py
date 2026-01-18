import collections
import contextlib
import copy
import inspect
import json
import sys
import textwrap
from typing import (
from itertools import zip_longest
from importlib.metadata import version as importlib_version
from typing import Final
import jsonschema
import jsonschema.exceptions
import jsonschema.validators
import numpy as np
import pandas as pd
from packaging.version import Version
from altair import vegalite
def _use_referencing_library() -> bool:
    """In version 4.18.0, the jsonschema package deprecated RefResolver in
    favor of the referencing library."""
    return Version(jsonschema_version_str) >= Version('4.18')