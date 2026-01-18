import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
class ElementSignatureTest(ComparisonTestCase):
    """
    Test that Element signatures are consistent.
    """

    def test_curve_string_signature(self):
        curve = Curve([], 'a', 'b')
        self.assertEqual(curve.kdims, [Dimension('a')])
        self.assertEqual(curve.vdims, [Dimension('b')])

    def test_area_string_signature(self):
        area = Area([], 'a', 'b')
        self.assertEqual(area.kdims, [Dimension('a')])
        self.assertEqual(area.vdims, [Dimension('b')])

    def test_errorbars_string_signature(self):
        errorbars = ErrorBars([], 'a', ['b', 'c'])
        self.assertEqual(errorbars.kdims, [Dimension('a')])
        self.assertEqual(errorbars.vdims, [Dimension('b'), Dimension('c')])

    def test_bars_string_signature(self):
        bars = Bars([], 'a', 'b')
        self.assertEqual(bars.kdims, [Dimension('a')])
        self.assertEqual(bars.vdims, [Dimension('b')])

    def test_boxwhisker_string_signature(self):
        boxwhisker = BoxWhisker([], 'a', 'b')
        self.assertEqual(boxwhisker.kdims, [Dimension('a')])
        self.assertEqual(boxwhisker.vdims, [Dimension('b')])

    def test_scatter_string_signature(self):
        scatter = Scatter([], 'a', 'b')
        self.assertEqual(scatter.kdims, [Dimension('a')])
        self.assertEqual(scatter.vdims, [Dimension('b')])

    def test_points_string_signature(self):
        points = Points([], ['a', 'b'], 'c')
        self.assertEqual(points.kdims, [Dimension('a'), Dimension('b')])
        self.assertEqual(points.vdims, [Dimension('c')])

    def test_vectorfield_string_signature(self):
        vectorfield = VectorField([], ['a', 'b'], ['c', 'd'])
        self.assertEqual(vectorfield.kdims, [Dimension('a'), Dimension('b')])
        self.assertEqual(vectorfield.vdims, [Dimension('c'), Dimension('d')])

    def test_vectorfield_from_uv(self):
        x = np.linspace(-1, 1, 4)
        X, Y = np.meshgrid(x, x)
        U, V = (3 * X, 4 * Y)
        vectorfield = VectorField.from_uv((X, Y, U, V))
        angle = np.arctan2(V, U)
        mag = np.hypot(U, V)
        kdims = [Dimension('x'), Dimension('y')]
        vdims = [Dimension('Angle', cyclic=True, range=(0, 2 * np.pi)), Dimension('Magnitude')]
        self.assertEqual(vectorfield.kdims, kdims)
        self.assertEqual(vectorfield.vdims, vdims)
        self.assertEqual(vectorfield.dimension_values(0), X.T.flatten())
        self.assertEqual(vectorfield.dimension_values(1), Y.T.flatten())
        self.assertEqual(vectorfield.dimension_values(2), angle.T.flatten())
        self.assertEqual(vectorfield.dimension_values(3), mag.T.flatten())

    def test_vectorfield_from_uv_dataframe(self):
        x = np.linspace(-1, 1, 4)
        X, Y = np.meshgrid(x, x)
        U, V = (5 * X, 5 * Y)
        df = pd.DataFrame({'x': X.flatten(), 'y': Y.flatten(), 'u': U.flatten(), 'v': V.flatten()})
        vectorfield = VectorField.from_uv(df, ['x', 'y'], ['u', 'v'])
        angle = np.arctan2(V, U)
        mag = np.hypot(U, V)
        kdims = [Dimension('x'), Dimension('y')]
        vdims = [Dimension('Angle', cyclic=True, range=(0, 2 * np.pi)), Dimension('Magnitude')]
        self.assertEqual(vectorfield.kdims, kdims)
        self.assertEqual(vectorfield.vdims, vdims)
        self.assertEqual(vectorfield.dimension_values(2, flat=False), angle.flat)
        self.assertEqual(vectorfield.dimension_values(3, flat=False), mag.flat)

    def test_path_string_signature(self):
        path = Path([], ['a', 'b'])
        self.assertEqual(path.kdims, [Dimension('a'), Dimension('b')])

    def test_spikes_string_signature(self):
        spikes = Spikes([], 'a')
        self.assertEqual(spikes.kdims, [Dimension('a')])

    def test_contours_string_signature(self):
        contours = Contours([], ['a', 'b'])
        self.assertEqual(contours.kdims, [Dimension('a'), Dimension('b')])

    def test_polygons_string_signature(self):
        polygons = Polygons([], ['a', 'b'])
        self.assertEqual(polygons.kdims, [Dimension('a'), Dimension('b')])

    def test_heatmap_string_signature(self):
        heatmap = HeatMap([], ['a', 'b'], 'c')
        self.assertEqual(heatmap.kdims, [Dimension('a'), Dimension('b')])
        self.assertEqual(heatmap.vdims, [Dimension('c')])

    def test_raster_string_signature(self):
        raster = Raster(np.array([[0]]), ['a', 'b'], 'c')
        self.assertEqual(raster.kdims, [Dimension('a'), Dimension('b')])
        self.assertEqual(raster.vdims, [Dimension('c')])

    def test_image_string_signature(self):
        img = Image(np.array([[0, 1], [0, 1]]), ['a', 'b'], 'c')
        self.assertEqual(img.kdims, [Dimension('a'), Dimension('b')])
        self.assertEqual(img.vdims, [Dimension('c')])

    def test_rgb_string_signature(self):
        img = RGB(np.zeros((2, 2, 3)), ['a', 'b'], ['R', 'G', 'B'])
        self.assertEqual(img.kdims, [Dimension('a'), Dimension('b')])
        self.assertEqual(img.vdims, [Dimension('R'), Dimension('G'), Dimension('B')])

    def test_quadmesh_string_signature(self):
        qmesh = QuadMesh(([0, 1], [0, 1], np.array([[0, 1], [0, 1]])), ['a', 'b'], 'c')
        self.assertEqual(qmesh.kdims, [Dimension('a'), Dimension('b')])
        self.assertEqual(qmesh.vdims, [Dimension('c')])