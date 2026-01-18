import numpy as np
import holoviews as hv
from holoviews.element.comparison import ComparisonTestCase
class TestEllipsisTable(ComparisonTestCase):

    def setUp(self):
        keys = [('M', 10), ('M', 16), ('F', 12)]
        values = [(15, 0.8), (18, 0.6), (10, 0.8)]
        self.table = hv.Table(zip(keys, values), kdims=['Gender', 'Age'], vdims=['Weight', 'Height'])
        super().setUp()

    def test_table_ellipsis_slice_value_weight(self):
        sliced = self.table[..., 'Weight']
        assert sliced.vdims == ['Weight']

    def test_table_ellipsis_slice_value_height(self):
        sliced = self.table[..., 'Height']
        assert sliced.vdims == ['Height']

    def test_table_ellipsis_slice_key_gender(self):
        sliced = self.table['M', ...]
        if not all((el == 'M' for el in sliced.dimension_values('Gender'))):
            raise AssertionError("Table key slicing on 'Gender' failed.")