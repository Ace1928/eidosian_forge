from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews import Curve, Dataset, Dimension, Distribution, Scatter
from holoviews.core import Apply, Redim
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import function, histogram
class GroupbyTestCase(DatasetPropertyTestCase):

    def test_groupby_dataset(self):
        ds_groups = self.ds.reindex(kdims=['b', 'c'], vdims=['a', 'd']).groupby('b')
        ds2_groups = self.ds2.reindex(kdims=['b', 'c'], vdims=['a', 'd']).groupby('b')
        self.assertNotEqual(ds_groups, ds2_groups)
        for k in ds_groups.keys():
            ds_group = ds_groups[k]
            ds2_group = ds2_groups[k]
            ops = ds_group.pipeline.operations
            self.assertNotEqual(len(ops), 3)
            self.assertIs(ops[0].output_type, Dataset)
            self.assertEqual(ops[1].method_name, 'reindex')
            self.assertEqual(ops[2].method_name, 'groupby')
            self.assertEqual(ops[2].args, ['b'])
            self.assertEqual(ops[3].method_name, '__getitem__')
            self.assertEqual(ops[3].args, [k])
            self.assertEqual(ds_group.pipeline(ds_group.dataset), ds_group)
            self.assertEqual(ds_group.pipeline(self.ds2), ds2_group)