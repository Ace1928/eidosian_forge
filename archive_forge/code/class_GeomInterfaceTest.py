from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset, MultiInterface
from holoviews.core.data.interface import DataError
from holoviews.element import Polygons, Path
from holoviews.element.comparison import ComparisonTestCase
from holoviews.tests.core.data.test_multiinterface import MultiBaseInterfaceTest
from geoviews.data.geom_dict import GeomDictInterface
class GeomInterfaceTest(ComparisonTestCase):
    """
    Test for the MultiInterface and GeomDictInterface.
    """
    __test__ = False

    def setUp(self):
        if sgeom is None:
            raise SkipTest('GeomInterfaceTest requires shapely, skipping tests')
        super().setUp()

    def test_multi_geom_dataset_geom_list_constructor(self):
        geoms = [sgeom.Polygon([(0, 0), (3, 3), (6, 0)])]
        Dataset(geoms, kdims=['x', 'y'], datatype=[self.datatype])

    def test_multi_geom_dataset_geom_dict_constructor(self):
        geoms = [{'geometry': sgeom.Polygon([(0, 0), (3, 3), (6, 0)])}]
        Dataset(geoms, kdims=['x', 'y'], datatype=[self.datatype])

    def test_multi_geom_dataset_geom_dict_constructor_extra_kdim(self):
        geoms = [{'geometry': sgeom.Polygon([(0, 0), (3, 3), (6, 0)]), 'z': 1}]
        Dataset(geoms, kdims=['x', 'y', 'z'], datatype=[self.datatype])

    def test_multi_geom_dataset_poly_coord_values(self):
        geoms = [sgeom.Polygon([(0, 0), (6, 6), (3, 3)])]
        mds = Dataset(geoms, kdims=['x', 'y'], datatype=[self.datatype])
        self.assertEqual(mds.dimension_values('x'), np.array([0, 6, 3, 0]))
        self.assertEqual(mds.dimension_values('y'), np.array([0, 6, 3, 0]))

    def test_multi_geom_dataset_poly_scalar_values(self):
        geoms = [{'geometry': sgeom.Polygon([(0, 0), (3, 3), (6, 0)]), 'z': 1}]
        mds = Dataset(geoms, kdims=['x', 'y', 'z'], datatype=[self.datatype])
        self.assertEqual(mds.dimension_values('z'), np.array([1, 1, 1, 1]))
        self.assertEqual(mds.dimension_values('z', expanded=False), np.array([1]))

    def test_multi_geom_poly_length(self):
        geoms = [{'geometry': sgeom.Polygon([(0, 0), (3, 3), (6, 0)])}, {'geometry': sgeom.Polygon([(3, 3), (9, 3), (6, 0)])}]
        mds = Dataset(geoms, kdims=['x', 'y'], datatype=[self.datatype])
        self.assertEqual(len(mds), 2)

    def test_multi_geom_poly_range(self):
        geoms = [{'geometry': sgeom.Polygon([(0, 0), (3, 3), (6, 0)])}, {'geometry': sgeom.Polygon([(3, 3), (9, 3), (6, 0)])}]
        mds = Dataset(geoms, kdims=['x', 'y'], datatype=[self.datatype])
        self.assertEqual(mds.range('x'), (0, 9))
        self.assertEqual(mds.range('y'), (0, 3))

    def test_multi_geom_dataset_line_string_coord_values(self):
        geoms = [sgeom.LineString([(0, 0), (3, 3), (6, 0)])]
        mds = Dataset(geoms, kdims=['x', 'y'], datatype=[self.datatype])
        self.assertEqual(mds.dimension_values('x'), np.array([0, 3, 6]))
        self.assertEqual(mds.dimension_values('y'), np.array([0, 3, 0]))

    def test_multi_geom_dataset_line_string_scalar_values(self):
        geoms = [{'geometry': sgeom.LineString([(0, 0), (3, 3), (6, 0)]), 'z': 1}]
        mds = Dataset(geoms, kdims=['x', 'y', 'z'], datatype=[self.datatype])
        self.assertEqual(mds.dimension_values('z'), np.array([1, 1, 1]))
        self.assertEqual(mds.dimension_values('z', expanded=False), np.array([1]))

    def test_multi_geom_line_string_length(self):
        geoms = [{'geometry': sgeom.LineString([(0, 0), (3, 3), (6, 0)])}, {'geometry': sgeom.LineString([(3, 3), (9, 3), (6, 0)])}]
        mds = Dataset(geoms, kdims=['x', 'y'], datatype=[self.datatype])
        self.assertEqual(len(mds), 2)

    def test_multi_geom_point_length(self):
        geoms = [{'geometry': sgeom.Point([(0, 0)])}, {'geometry': sgeom.Point([(3, 3)])}]
        mds = Dataset(geoms, kdims=['x', 'y'], datatype=[self.datatype])
        self.assertEqual(len(mds), 2)

    def test_multi_geom_point_coord_values(self):
        geoms = [{'geometry': sgeom.Point([(0, 1)])}, {'geometry': sgeom.Point([(3, 5)])}]
        mds = Dataset(geoms, kdims=['x', 'y'], datatype=[self.datatype])
        self.assertEqual(mds.dimension_values('x'), np.array([0, 3]))
        self.assertEqual(mds.dimension_values('y'), np.array([1, 5]))

    def test_multi_geom_point_coord_range(self):
        geoms = [{'geometry': sgeom.Point([(0, 1)])}, {'geometry': sgeom.Point([(3, 5)])}]
        mds = Dataset(geoms, kdims=['x', 'y'], datatype=[self.datatype])
        self.assertEqual(mds.range('x'), (0, 3))
        self.assertEqual(mds.range('y'), (1, 5))

    def test_multi_dict_groupby(self):
        geoms = [{'geometry': sgeom.Polygon([(2, 0), (1, 2), (0, 0)]), 'z': 1}, {'geometry': sgeom.Polygon([(3, 3), (3, 3), (6, 0)]), 'z': 2}]
        mds = Dataset(geoms, kdims=['x', 'y', 'z'], datatype=[self.datatype])
        for i, (k, ds) in enumerate(mds.groupby('z').items()):
            self.assertEqual(k, geoms[i]['z'])
            self.assertEqual(ds.clone(vdims=[]), Dataset([geoms[i]], kdims=['x', 'y']))