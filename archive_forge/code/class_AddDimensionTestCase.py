from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews import Curve, Dataset, Dimension, Distribution, Scatter
from holoviews.core import Apply, Redim
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import function, histogram
class AddDimensionTestCase(DatasetPropertyTestCase):

    def test_add_dimension_dataset(self):
        ds_dim_added = self.ds.add_dimension('new', 1, 17)
        ds2_dim_added = self.ds2.add_dimension('new', 1, 17)
        self.assertNotEqual(ds_dim_added, ds2_dim_added)
        self.assertEqual(ds_dim_added.dataset, self.ds)
        self.assertEqual(ds2_dim_added.dataset, self.ds2)
        ops = ds_dim_added.pipeline.operations
        self.assertEqual(len(ops), 2)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertEqual(ops[1].method_name, 'add_dimension')
        self.assertEqual(ops[1].args, ['new', 1, 17])
        self.assertEqual(ops[1].kwargs, {})
        self.assertEqual(ds_dim_added.pipeline(ds_dim_added.dataset), ds_dim_added)
        self.assertEqual(ds_dim_added.pipeline(self.ds2), ds2_dim_added)