import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
def _skip_test_for_braket(dev):
    """Skip the specific test because the Braket plugin does not yet support custom measurement processes."""
    if 'braket' in getattr(dev, 'short_name', dev.name):
        pytest.skip(f'Custom measurement test skipped for {dev.short_name}.')