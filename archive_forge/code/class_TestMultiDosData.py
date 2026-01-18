from collections import OrderedDict
import numpy as np
import pytest
from typing import List, Tuple, Any
from ase.spectrum.dosdata import DOSData, GridDOSData, RawDOSData
class TestMultiDosData:
    """Test interaction between DOS data objects"""

    @pytest.fixture
    def sparse_dos(self):
        return RawDOSData([1.2, 3.4, 5.0], [3.0, 2.1, 0.0], info={'symbol': 'H', 'number': '1', 'food': 'egg'})

    @pytest.fixture
    def dense_dos(self):
        x = np.linspace(0.0, 10.0, 11)
        y = np.sin(x / 10)
        return GridDOSData(x, y, info={'symbol': 'C', 'orbital': '2s', 'day': 'Tue'})

    def test_addition(self, sparse_dos, dense_dos):
        with pytest.raises(TypeError):
            sparse_dos + dense_dos
        with pytest.raises(TypeError):
            dense_dos + sparse_dos