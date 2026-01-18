from collections import OrderedDict
import numpy as np
import pytest
from typing import List, Tuple, Any
from ase.spectrum.dosdata import DOSData, GridDOSData, RawDOSData
class TestDosData:
    """Test the abstract base class for DOS data"""
    sample_info = [(None, {}), ({}, {}), ({'symbol': 'C', 'index': '2', 'strangekey': 'isallowed'}, {'symbol': 'C', 'index': '2', 'strangekey': 'isallowed'}), ('notadict', TypeError), (False, TypeError)]

    @pytest.mark.parametrize('info, expected', sample_info)
    def test_dosdata_init_info(self, info, expected):
        """Check 'info' parameter is handled properly"""
        if isinstance(expected, type) and isinstance(expected(), Exception):
            with pytest.raises(expected):
                dos_data = MinimalDOSData(info=info)
        else:
            dos_data = MinimalDOSData(info=info)
            assert dos_data.info == expected

    @pytest.mark.parametrize('info, expected', [({}, ''), ({'key1': 'value1'}, 'key1: value1'), (OrderedDict([('key1', 'value1'), ('key2', 'value2')]), 'key1: value1; key2: value2'), ({'key1': 'value1', 'label': 'xyz'}, 'xyz'), ({'label': 'xyz'}, 'xyz')])
    def test_label_from_info(self, info, expected):
        assert DOSData.label_from_info(info) == expected