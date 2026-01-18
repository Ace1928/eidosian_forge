import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
class GriddedInterfaceTests:
    """
    Tests for the grid interfaces
    """
    __test__ = False

    def init_grid_data(self):
        self.grid_xs = np.array([0, 1])
        self.grid_ys = np.array([0.1, 0.2, 0.3])
        self.grid_zs = np.array([[0, 1], [2, 3], [4, 5]])
        self.dataset_grid = self.element((self.grid_xs, self.grid_ys, self.grid_zs), ['x', 'y'], ['z'])
        self.dataset_grid_alias = self.element((self.grid_xs, self.grid_ys, self.grid_zs), [('x', 'X'), ('y', 'Y')], [('z', 'Z')])
        self.dataset_grid_inv = self.element((self.grid_xs[::-1], self.grid_ys[::-1], self.grid_zs), ['x', 'y'], ['z'])

    def test_canonical_vdim(self):
        x = np.array([0.0, 0.75, 1.5])
        y = np.array([1.5, 0.75, 0.0])
        z = np.array([[0.06925999, 0.05800389, 0.05620127], [0.06240918, 0.05800931, 0.04969735], [0.05376789, 0.04669417, 0.03880118]])
        dataset = self.element((x, y, z), kdims=['x', 'y'], vdims=['z'])
        canonical = np.array([[0.05376789, 0.04669417, 0.03880118], [0.06240918, 0.05800931, 0.04969735], [0.06925999, 0.05800389, 0.05620127]])
        self.assertEqual(dataset.dimension_values('z', flat=False), canonical)

    def test_gridded_dtypes(self):
        ds = self.dataset_grid
        self.assertEqual(ds.interface.dtype(ds, 'x'), np.dtype(int))
        self.assertEqual(ds.interface.dtype(ds, 'y'), np.float64)
        self.assertEqual(ds.interface.dtype(ds, 'z'), np.dtype(int))

    def test_select_slice(self):
        ds = self.element((self.grid_xs, self.grid_ys[:2], self.grid_zs[:2]), ['x', 'y'], ['z'])
        self.assertEqual(self.dataset_grid.select(y=slice(0, 0.25)), ds)

    def test_select_tuple(self):
        ds = self.element((self.grid_xs, self.grid_ys[:2], self.grid_zs[:2]), ['x', 'y'], ['z'])
        self.assertEqual(self.dataset_grid.select(y=(0, 0.25)), ds)

    def test_nodata_range(self):
        ds = self.dataset_grid.clone(vdims=[Dimension('z', nodata=0)])
        self.assertEqual(ds.range('z'), (1, 5))

    def test_dataset_ndloc_index(self):
        xs, ys = (np.linspace(0.12, 0.81, 10), np.linspace(0.12, 0.391, 5))
        arr = np.arange(10) * np.arange(5)[np.newaxis].T
        ds = self.element((xs, ys, arr), kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype])
        self.assertEqual(ds.ndloc[0, 0], arr[0, 0])

    def test_dataset_ndloc_index_inverted_x(self):
        xs, ys = (np.linspace(0.12, 0.81, 10), np.linspace(0.12, 0.391, 5))
        arr = np.arange(10) * np.arange(5)[np.newaxis].T
        ds = self.element((xs[::-1], ys, arr), kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype])
        self.assertEqual(ds.ndloc[0, 0], arr[0, 9])

    def test_dataset_ndloc_index_inverted_y(self):
        xs, ys = (np.linspace(0.12, 0.81, 10), np.linspace(0.12, 0.391, 5))
        arr = np.arange(10) * np.arange(5)[np.newaxis].T
        ds = self.element((xs, ys[::-1], arr), kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype])
        self.assertEqual(ds.ndloc[0, 0], arr[4, 0])

    def test_dataset_ndloc_index_inverted_xy(self):
        xs, ys = (np.linspace(0.12, 0.81, 10), np.linspace(0.12, 0.391, 5))
        arr = np.arange(10) * np.arange(5)[np.newaxis].T
        ds = self.element((xs[::-1], ys[::-1], arr), kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype])
        self.assertEqual(ds.ndloc[0, 0], arr[4, 9])

    def test_dataset_ndloc_index2(self):
        xs, ys = (np.linspace(0.12, 0.81, 10), np.linspace(0.12, 0.391, 5))
        arr = np.arange(10) * np.arange(5)[np.newaxis].T
        ds = self.element((xs, ys, arr), kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype])
        self.assertEqual(ds.ndloc[4, 9], arr[4, 9])

    def test_dataset_ndloc_slice(self):
        xs, ys = (np.linspace(0.12, 0.81, 10), np.linspace(0.12, 0.391, 5))
        arr = np.arange(10) * np.arange(5)[np.newaxis].T
        ds = self.element((xs, ys, arr), kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype])
        sliced = self.element((xs[2:5], ys[1:], arr[1:, 2:5]), kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype])
        self.assertEqual(ds.ndloc[1:, 2:5], sliced)

    def test_dataset_ndloc_slice_inverted_x(self):
        xs, ys = (np.linspace(0.12, 0.81, 10), np.linspace(0.12, 0.391, 5))
        arr = np.arange(10) * np.arange(5)[np.newaxis].T
        ds = self.element((xs[::-1], ys, arr), kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype])
        sliced = self.element((xs[::-1][5:8], ys[1:], arr[1:, 5:8]), kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype])
        self.assertEqual(ds.ndloc[1:, 2:5], sliced)

    def test_dataset_ndloc_slice_inverted_y(self):
        xs, ys = (np.linspace(0.12, 0.81, 10), np.linspace(0.12, 0.391, 5))
        arr = np.arange(10) * np.arange(5)[np.newaxis].T
        ds = self.element((xs, ys[::-1], arr), kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype])
        sliced = self.element((xs[2:5], ys[::-1][:-1], arr[:-1, 2:5]), kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype])
        self.assertEqual(ds.ndloc[1:, 2:5], sliced)

    def test_dataset_ndloc_slice_inverted_xy(self):
        xs, ys = (np.linspace(0.12, 0.81, 10), np.linspace(0.12, 0.391, 5))
        arr = np.arange(10) * np.arange(5)[np.newaxis].T
        ds = self.element((xs[::-1], ys[::-1], arr), kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype])
        sliced = self.element((xs[::-1][5:8], ys[::-1][:-1], arr[:-1, 5:8]), kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype])
        self.assertEqual(ds.ndloc[1:, 2:5], sliced)

    def test_dataset_dim_vals_grid_kdims_xs(self):
        self.assertEqual(self.dataset_grid.dimension_values(0, expanded=False), np.array([0, 1]))

    def test_dataset_dim_vals_grid_kdims_xs_alias(self):
        self.assertEqual(self.dataset_grid_alias.dimension_values('x', expanded=False), np.array([0, 1]))
        self.assertEqual(self.dataset_grid_alias.dimension_values('X', expanded=False), np.array([0, 1]))

    def test_dataset_dim_vals_grid_kdims_xs_inv(self):
        self.assertEqual(self.dataset_grid_inv.dimension_values(0, expanded=False), np.array([0, 1]))

    def test_dataset_dim_vals_grid_kdims_expanded_xs_flat(self):
        expanded_xs = np.array([0, 0, 0, 1, 1, 1])
        self.assertEqual(self.dataset_grid.dimension_values(0), expanded_xs)

    def test_dataset_dim_vals_grid_kdims_expanded_xs_flat_inv(self):
        expanded_xs = np.array([0, 0, 0, 1, 1, 1])
        self.assertEqual(self.dataset_grid_inv.dimension_values(0), expanded_xs)

    def test_dataset_dim_vals_grid_kdims_expanded_xs(self):
        expanded_xs = np.array([[0, 1], [0, 1], [0, 1]])
        self.assertEqual(self.dataset_grid.dimension_values(0, flat=False), expanded_xs)

    def test_dataset_dim_vals_grid_kdims_expanded_xs_inv(self):
        expanded_xs = np.array([[0, 1], [0, 1], [0, 1]])
        self.assertEqual(self.dataset_grid_inv.dimension_values(0, flat=False), expanded_xs)

    def test_dataset_dim_vals_grid_kdims_ys(self):
        self.assertEqual(self.dataset_grid.dimension_values(1, expanded=False), np.array([0.1, 0.2, 0.3]))

    def test_dataset_dim_vals_grid_kdims_ys_inv(self):
        self.assertEqual(self.dataset_grid_inv.dimension_values(1, expanded=False), np.array([0.1, 0.2, 0.3]))

    def test_dataset_dim_vals_grid_kdims_expanded_ys_flat(self):
        expanded_ys = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3])
        self.assertEqual(self.dataset_grid.dimension_values(1), expanded_ys)

    def test_dataset_dim_vals_grid_kdims_expanded_ys_flat_inv(self):
        expanded_ys = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3])
        self.assertEqual(self.dataset_grid_inv.dimension_values(1), expanded_ys)

    def test_dataset_dim_vals_grid_kdims_expanded_ys(self):
        expanded_ys = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
        self.assertEqual(self.dataset_grid.dimension_values(1, flat=False), expanded_ys)

    def test_dataset_dim_vals_grid_kdims_expanded_ys_inv(self):
        expanded_ys = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
        self.assertEqual(self.dataset_grid_inv.dimension_values(1, flat=False), expanded_ys)

    def test_dataset_dim_vals_dimensions_match_shape(self):
        self.assertEqual(len({self.dataset_grid.dimension_values(i, flat=False).shape for i in range(3)}), 1)

    def test_dataset_dim_vals_dimensions_match_shape_inv(self):
        self.assertEqual(len({self.dataset_grid_inv.dimension_values(i, flat=False).shape for i in range(3)}), 1)

    def test_dataset_dim_vals_grid_vdims_zs_flat(self):
        expanded_zs = np.array([0, 2, 4, 1, 3, 5])
        self.assertEqual(self.dataset_grid.dimension_values(2), expanded_zs)

    def test_dataset_dim_vals_grid_vdims_zs_flat_inv(self):
        expanded_zs = np.array([5, 3, 1, 4, 2, 0])
        self.assertEqual(self.dataset_grid_inv.dimension_values(2), expanded_zs)

    def test_dataset_dim_vals_grid_vdims_zs(self):
        expanded_zs = np.array([[0, 1], [2, 3], [4, 5]])
        self.assertEqual(self.dataset_grid.dimension_values(2, flat=False), expanded_zs)

    def test_dataset_dim_vals_grid_vdims_zs_inv(self):
        expanded_zs = np.array([[5, 4], [3, 2], [1, 0]])
        self.assertEqual(self.dataset_grid_inv.dimension_values(2, flat=False), expanded_zs)

    def test_dataset_groupby_with_transposed_dimensions(self):
        dat = np.zeros((3, 5, 7))
        dataset = Dataset((range(7), range(5), range(3), dat), ['z', 'x', 'y'], 'value')
        grouped = dataset.groupby('z', kdims=['y', 'x'])
        self.assertEqual(grouped.last.dimension_values(2, flat=False), dat[:, :, -1].T)

    def test_dataset_dynamic_groupby_with_transposed_dimensions(self):
        dat = np.zeros((3, 5, 7))
        dataset = Dataset((range(7), range(5), range(3), dat), ['z', 'x', 'y'], 'value')
        grouped = dataset.groupby('z', kdims=['y', 'x'], dynamic=True)
        self.assertEqual(grouped[2].dimension_values(2, flat=False), dat[:, :, -1].T)

    def test_dataset_slice_inverted_dimension(self):
        xs = np.arange(30)[::-1]
        ys = np.random.rand(30)
        ds = Dataset((xs, ys), 'x', 'y')
        sliced = ds[5:15]
        self.assertEqual(sliced, Dataset((xs[15:25], ys[15:25]), 'x', 'y'))

    def test_sample_2d(self):
        xs = ys = np.linspace(0, 6, 50)
        XS, YS = np.meshgrid(xs, ys)
        values = np.sin(XS)
        sampled = Dataset((xs, ys, values), ['x', 'y'], 'z').sample(y=0)
        self.assertEqual(sampled, Curve((xs, values[0]), vdims='z'))

    def test_aggregate_2d_with_spreadfn(self):
        array = np.random.rand(10, 5)
        ds = Dataset((range(5), range(10), array), ['x', 'y'], 'z')
        agg = ds.aggregate('x', np.mean, np.std)
        example = Dataset((range(5), array.mean(axis=0), array.std(axis=0)), 'x', ['z', 'z_std'])
        self.assertEqual(agg, example)

    def test_concat_grid_3d(self):
        array = np.random.rand(4, 5, 3, 2)
        orig = Dataset((range(2), range(3), range(5), range(4), array), ['A', 'B', 'x', 'y'], 'z')
        hmap = HoloMap({(i, j): self.element((range(5), range(4), array[:, :, j, i]), ['x', 'y'], 'z') for i in range(2) for j in range(3)}, ['A', 'B'])
        ds = concat(hmap)
        self.assertEqual(ds, orig)

    def test_concat_grid_3d_shape_mismatch(self):
        ds1 = Dataset(([0, 1], [1, 2, 3], np.random.rand(3, 2)), ['x', 'y'], 'z')
        ds2 = Dataset(([0, 1, 2], [1, 2], np.random.rand(2, 3)), ['x', 'y'], 'z')
        hmap = HoloMap({1: ds1, 2: ds2})
        with self.assertRaises(DataError):
            concat(hmap)

    def test_grid_3d_groupby_concat_roundtrip(self):
        array = np.random.rand(4, 5, 3, 2)
        orig = Dataset((range(2), range(3), range(5), range(4), array), ['A', 'B', 'x', 'y'], 'z')
        self.assertEqual(concat(orig.groupby(['A', 'B'])), orig)