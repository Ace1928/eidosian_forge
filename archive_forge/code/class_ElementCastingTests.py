import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
class ElementCastingTests(ComparisonTestCase):
    """
    Tests whether casting an element will faithfully copy data and
    parameters. Important to check for elements where data is not all
    held on .data attribute, e.g. Image bounds or Graph nodes and
    edgepaths.
    """

    def test_image_casting(self):
        img = Image([], bounds=2)
        self.assertEqual(img, Image(img))

    def test_rgb_casting(self):
        rgb = RGB([], bounds=2)
        self.assertEqual(rgb, RGB(rgb))

    def test_graph_casting(self):
        graph = Graph(([(0, 1)], [(0, 0, 0), (0, 1, 1)]))
        self.assertEqual(graph, Graph(graph))

    def test_trimesh_casting(self):
        trimesh = TriMesh(([(0, 1, 2)], [(0, 0, 0), (0, 1, 1), (1, 1, 2)]))
        self.assertEqual(trimesh, TriMesh(trimesh))