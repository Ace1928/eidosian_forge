from holoviews import Overlay
from holoviews.annotators import PathAnnotator, PointAnnotator, annotate
from holoviews.element import Path, Points, Table
from holoviews.element.tiles import EsriStreet, Tiles
from holoviews.tests.plotting.bokeh.test_plot import TestBokehPlot
class Test_annotate(TestBokehPlot):

    def test_compose_annotators(self):
        layout1 = annotate(Points([]), annotations=['Label'])
        layout2 = annotate(Path([]), annotations=['Name'])
        combined = annotate.compose(layout1, layout2)
        overlay = combined.DynamicMap.I[()]
        tables = combined.Annotator.I[()]
        self.assertIsInstance(overlay, Overlay)
        self.assertEqual(len(overlay), 2)
        self.assertEqual(overlay.get(0), Points([], vdims='Label'))
        self.assertEqual(overlay.get(1), Path([], vdims='Name'))
        self.assertIsInstance(tables, Overlay)
        self.assertEqual(len(tables), 3)

    def test_annotate_overlay(self):
        layout = annotate(EsriStreet() * Points([]), annotations=['Label'])
        overlay = layout.DynamicMap.I[()]
        tables = layout.Annotator.PointAnnotator[()]
        self.assertIsInstance(overlay, Overlay)
        self.assertEqual(len(overlay), 2)
        self.assertIsInstance(overlay.get(0), Tiles)
        self.assertEqual(overlay.get(1), Points([], vdims='Label'))
        self.assertIsInstance(tables, Overlay)
        self.assertEqual(len(tables), 1)

    def test_annotated_property(self):
        annotator = annotate.instance()
        annotator(Points([]), annotations=['Label'])
        self.assertIn('Label', annotator.annotated)

    def test_selected_property(self):
        annotator = annotate.instance()
        annotator(Points([(1, 2), (2, 3)]), annotations=['Label'])
        annotator.annotator._selection.update(index=[1])
        self.assertEqual(annotator.selected, Points([(2, 3, '')], vdims='Label'))