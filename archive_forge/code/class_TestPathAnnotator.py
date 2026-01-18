from holoviews import Overlay
from holoviews.annotators import PathAnnotator, PointAnnotator, annotate
from holoviews.element import Path, Points, Table
from holoviews.element.tiles import EsriStreet, Tiles
from holoviews.tests.plotting.bokeh.test_plot import TestBokehPlot
class TestPathAnnotator(TestBokehPlot):

    def test_add_annotations(self):
        annotator = PathAnnotator(Path([]), annotations=['Label'])
        self.assertIn('Label', annotator.object)

    def test_add_name(self):
        annotator = PathAnnotator(name='Test Annotator', annotations=['Label'])
        self.assertEqual(annotator._stream.tooltip, 'Test Annotator Tool')
        self.assertEqual(annotator._vertex_stream.tooltip, 'Test Annotator Edit Tool')
        self.assertEqual(annotator._table.label, 'Test Annotator')
        self.assertEqual(annotator._vertex_table.label, 'Test Annotator Vertices')
        self.assertEqual(annotator.editor._names, ['Test Annotator', 'Test Annotator Vertices'])

    def test_add_vertex_annotations(self):
        annotator = PathAnnotator(Path([]), vertex_annotations=['Label'])
        self.assertIn('Label', annotator.object)

    def test_replace_object(self):
        annotator = PathAnnotator(Path([]), annotations=['Label'], vertex_annotations=['Value'])
        annotator.object = Path([(1, 2), (2, 3), (0, 0)])
        self.assertIn('Label', annotator.object)
        expected = Table([''], kdims=['Label'], label='PathAnnotator')
        self.assertEqual(annotator._table, expected)
        expected = Table([], ['x', 'y'], 'Value', label='PathAnnotator Vertices')
        self.assertEqual(annotator._vertex_table, expected)
        self.assertIs(annotator._link.target, annotator._table)
        self.assertIs(annotator._vertex_link.target, annotator._vertex_table)