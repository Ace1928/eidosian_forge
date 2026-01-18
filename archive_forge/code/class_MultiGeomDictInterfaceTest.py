from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset, MultiInterface
from holoviews.core.data.interface import DataError
from holoviews.element import Polygons, Path
from holoviews.element.comparison import ComparisonTestCase
from holoviews.tests.core.data.test_multiinterface import MultiBaseInterfaceTest
from geoviews.data.geom_dict import GeomDictInterface
class MultiGeomDictInterfaceTest(MultiBaseInterfaceTest):
    datatype = 'multitabular'
    interface = MultiInterface
    subtype = 'geom_dictionary'
    __test__ = True

    def test_dict_dataset(self):
        dicts = [{'x': np.arange(i, i + 2), 'y': np.arange(i, i + 2)} for i in range(2)]
        mds = Path(dicts, kdims=['x', 'y'], datatype=[self.datatype])
        self.assertIs(mds.interface, self.interface)
        for i, cols in enumerate(mds.split(datatype='columns')):
            self.assertEqual(dict(cols), dict(dicts[i], geom_type='Line', geometry=mds.data[i]['geometry']))

    def test_polygon_dtype(self):
        poly = Polygons([{'x': [1, 2, 3], 'y': [2, 0, 7]}], datatype=[self.datatype])
        self.assertIs(poly.interface, self.interface)
        self.assertEqual(poly.interface.dtype(poly, 'x'), 'float64')

    def test_array_points_iloc_index_rows_index_cols(self):
        arrays = [np.array([(1 + i, i), (2 + i, i), (3 + i, i)]) for i in range(2)]
        mds = Dataset(arrays, kdims=['x', 'y'], datatype=[self.datatype])
        self.assertIs(mds.interface, self.interface)
        with self.assertRaises(DataError):
            mds.iloc[3, 0]

    def test_df_dataset(self):
        if not pd:
            raise SkipTest('Pandas not available')
        dfs = [pd.DataFrame(np.column_stack([np.arange(i, i + 2), np.arange(i, i + 2)]), columns=['x', 'y']) for i in range(2)]
        mds = Path(dfs, kdims=['x', 'y'], datatype=[self.datatype])
        self.assertIs(mds.interface, self.interface)
        for i, ds in enumerate(mds.split(datatype='dataframe')):
            ds['x'] = ds.x.astype(int)
            ds['y'] = ds.y.astype(int)
            self.assertEqual(ds, dfs[i])