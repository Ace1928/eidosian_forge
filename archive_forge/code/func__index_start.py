from datetime import datetime
import numpy as np
import pytest
from pandas import (
@pytest.fixture
def _index_start():
    """Fixture for parametrization of index, series and frame."""
    return datetime(2005, 1, 1)