import locale
import logging
import os
import pprint
import re
import sys
import warnings
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Dict
from typing_extensions import Literal
import numpy as np
def _add_none_to_validator(base_validator):
    """Create a validator function that catches none and then calls base_fun."""

    def validate_with_none(value):
        if value is None or (isinstance(value, str) and value.lower() == 'none'):
            return None
        return base_validator(value)
    return validate_with_none