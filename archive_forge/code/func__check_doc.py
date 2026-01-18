import json
from textwrap import dedent, indent
from unittest.mock import Mock, patch
import numpy as np
import pandas
import pytest
import modin.pandas as pd
import modin.utils
from modin.error_message import ErrorMessage
from modin.tests.pandas.utils import create_test_dfs
def _check_doc(wrapped, orig):
    assert wrapped.__doc__ == orig.__doc__
    if isinstance(wrapped, property):
        assert wrapped.fget.__doc_inherited__
    else:
        assert wrapped.__doc_inherited__