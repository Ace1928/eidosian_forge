from unittest import skip, skipIf
import pandas as pd
import panel as pn
import holoviews as hv
from holoviews.core.options import Cycle, Store
from holoviews.element import ErrorBars, Points, Rectangles, Table, VSpan
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.util import linear_gradient
from holoviews.selection import link_selections
from holoviews.streams import SelectionXY
class TestLinkSelections(ComparisonTestCase):
    __test__ = False

    def setUp(self):
        self.data = pd.DataFrame({'x': [1, 2, 3], 'y': [0, 3, 2], 'e': [1, 1.5, 2]}, columns=['x', 'y', 'e'])

    def element_color(self, element):
        raise NotImplementedError

    def check_base_points_like(self, base_points, lnk_sel, data=None):
        if data is None:
            data = self.data
        self.assertEqual(self.element_color(base_points), lnk_sel.unselected_color)
        self.assertEqual(base_points.data, data)

    @staticmethod
    def get_value_with_key_type(d, hvtype):
        for k, v in d.items():
            if isinstance(k, hvtype) or (isinstance(k, hv.DynamicMap) and k.type == hvtype):
                return v
        raise KeyError(f'No key with type {hvtype}')

    @staticmethod
    def expected_selection_color(element, lnk_sel):
        if lnk_sel.selected_color is not None:
            expected_color = lnk_sel.selected_color
        else:
            expected_color = element.opts.get(group='style')[0].get('color')
        return expected_color

    def check_overlay_points_like(self, overlay_points, lnk_sel, data):
        self.assertEqual(self.element_color(overlay_points), self.expected_selection_color(overlay_points, lnk_sel))
        self.assertEqual(overlay_points.data, data)

    def test_points_selection(self, dynamic=False, show_regions=True):
        points = Points(self.data)
        if dynamic:
            points = hv.util.Dynamic(points)
        lnk_sel = link_selections.instance(show_regions=show_regions, unselected_color='#ff0000')
        linked = lnk_sel(points)
        current_obj = linked[()]
        self.assertIsInstance(current_obj, hv.Overlay)
        unselected, selected, region, region2 = current_obj.values()
        self.check_base_points_like(unselected, lnk_sel)
        self.check_overlay_points_like(selected, lnk_sel, self.data)
        selectionxy = TestLinkSelections.get_value_with_key_type(lnk_sel._selection_expr_streams, hv.Points).input_streams[0].input_stream.input_streams[0]
        self.assertIsInstance(selectionxy, hv.streams.SelectionXY)
        selectionxy.event(bounds=(0, 1, 5, 5))
        unselected, selected, region, region2 = linked[()].values()
        self.check_base_points_like(unselected, lnk_sel)
        self.check_overlay_points_like(selected, lnk_sel, self.data.iloc[1:])
        if show_regions:
            self.assertEqual(region, Rectangles([(0, 1, 5, 5)]))
        else:
            self.assertEqual(region, Rectangles([]))

    def test_points_selection_hide_region(self):
        self.test_points_selection(show_regions=False)

    def test_points_selection_dynamic(self):
        self.test_points_selection(dynamic=True)

    def test_layout_selection_points_table(self):
        points = Points(self.data)
        table = Table(self.data)
        lnk_sel = link_selections.instance(selected_color='#aa0000', unselected_color='#ff0000')
        linked = lnk_sel(points + table)
        current_obj = linked[()]
        self.check_base_points_like(current_obj[0][()].Points.I, lnk_sel)
        self.check_overlay_points_like(current_obj[0][()].Points.II, lnk_sel, self.data)
        self.assertEqual(self.element_color(current_obj[1][()]), [lnk_sel.selected_color] * len(self.data))
        selectionxy = TestLinkSelections.get_value_with_key_type(lnk_sel._selection_expr_streams, hv.Points).input_streams[0].input_stream.input_streams[0]
        selectionxy.event(bounds=(0, 0, 4, 2))
        current_obj = linked[()]
        self.check_base_points_like(current_obj[0][()].Points.I, lnk_sel)
        self.check_overlay_points_like(current_obj[0][()].Points.II, lnk_sel, self.data.iloc[[0, 2]])
        self.assertEqual(self.element_color(current_obj[1][()]), [lnk_sel.selected_color, lnk_sel.unselected_color, lnk_sel.selected_color])

    def test_overlay_points_errorbars(self, dynamic=False):
        points = Points(self.data)
        error = ErrorBars(self.data, kdims='x', vdims=['y', 'e'])
        lnk_sel = link_selections.instance(unselected_color='#ff0000')
        overlay = points * error
        if dynamic:
            overlay = hv.util.Dynamic(overlay)
        linked = lnk_sel(overlay)
        current_obj = linked[()]
        self.check_base_points_like(current_obj.Points.I, lnk_sel)
        self.check_base_points_like(current_obj.ErrorBars.I, lnk_sel)
        self.check_overlay_points_like(current_obj.Points.II, lnk_sel, self.data)
        self.check_overlay_points_like(current_obj.ErrorBars.II, lnk_sel, self.data)
        selectionxy = TestLinkSelections.get_value_with_key_type(lnk_sel._selection_expr_streams, hv.Points).input_streams[0].input_stream.input_streams[0]
        selectionxy.event(bounds=(0, 0, 4, 2))
        current_obj = linked[()]
        self.check_base_points_like(current_obj.Points.I, lnk_sel)
        self.check_base_points_like(current_obj.ErrorBars.I, lnk_sel)
        self.check_overlay_points_like(current_obj.Points.II, lnk_sel, self.data.iloc[[0, 2]])
        self.check_overlay_points_like(current_obj.ErrorBars.II, lnk_sel, self.data.iloc[[0, 2]])

    @ds_skip
    def test_datashade_selection(self):
        points = Points(self.data)
        layout = points + dynspread(datashade(points))
        lnk_sel = link_selections.instance(unselected_color='#ff0000')
        linked = lnk_sel(layout)
        current_obj = linked[()]
        self.check_base_points_like(current_obj[0][()].Points.I, lnk_sel)
        self.check_overlay_points_like(current_obj[0][()].Points.II, lnk_sel, self.data)
        self.assertEqual(current_obj[1][()].RGB.I, dynspread(datashade(points, cmap=lnk_sel.unselected_cmap, alpha=255))[()])
        self.assertEqual(current_obj[1][()].RGB.II, dynspread(datashade(points, cmap=lnk_sel.selected_cmap, alpha=255))[()])
        selectionxy = TestLinkSelections.get_value_with_key_type(lnk_sel._selection_expr_streams, hv.Points).input_streams[0].input_stream.input_streams[0]
        self.assertIsInstance(selectionxy, SelectionXY)
        selectionxy.event(bounds=(0, 1, 5, 5))
        current_obj = linked[()]
        self.check_base_points_like(current_obj[0][()].Points.I, lnk_sel)
        self.check_overlay_points_like(current_obj[0][()].Points.II, lnk_sel, self.data.iloc[1:])
        self.assertEqual(current_obj[1][()].RGB.I, dynspread(datashade(points, cmap=lnk_sel.unselected_cmap, alpha=255))[()])
        self.assertEqual(current_obj[1][()].RGB.II, dynspread(datashade(points.iloc[1:], cmap=lnk_sel.selected_cmap, alpha=255))[()])

    @ds_skip
    def test_datashade_in_overlay_selection(self):
        points = Points(self.data)
        layout = points * dynspread(datashade(points))
        lnk_sel = link_selections.instance(unselected_color='#ff0000')
        linked = lnk_sel(layout)
        current_obj = linked[()]
        self.check_base_points_like(current_obj[()].Points.I, lnk_sel)
        self.check_overlay_points_like(current_obj[()].Points.II, lnk_sel, self.data)
        self.assertEqual(current_obj[()].RGB.I, dynspread(datashade(points, cmap=lnk_sel.unselected_cmap, alpha=255))[()])
        self.assertEqual(current_obj[()].RGB.II, dynspread(datashade(points, cmap=lnk_sel.selected_cmap, alpha=255))[()])
        selectionxy = TestLinkSelections.get_value_with_key_type(lnk_sel._selection_expr_streams, hv.Points).input_streams[0].input_stream.input_streams[0]
        self.assertIsInstance(selectionxy, SelectionXY)
        selectionxy.event(bounds=(0, 1, 5, 5))
        current_obj = linked[()]
        self.check_base_points_like(current_obj[()].Points.I, lnk_sel)
        self.check_overlay_points_like(current_obj[()].Points.II, lnk_sel, self.data.iloc[1:])
        self.assertEqual(current_obj[()].RGB.I, dynspread(datashade(points, cmap=lnk_sel.unselected_cmap, alpha=255))[()])
        self.assertEqual(current_obj[()].RGB.II, dynspread(datashade(points.iloc[1:], cmap=lnk_sel.selected_cmap, alpha=255))[()])

    def test_points_selection_streaming(self):
        buffer = hv.streams.Buffer(self.data.iloc[:2], index=False)
        points = hv.DynamicMap(Points, streams=[buffer])
        lnk_sel = link_selections.instance(unselected_color='#ff0000')
        linked = lnk_sel(points)
        selectionxy = TestLinkSelections.get_value_with_key_type(lnk_sel._selection_expr_streams, hv.Points).input_streams[0].input_stream.input_streams[0]
        self.assertIsInstance(selectionxy, hv.streams.SelectionXY)
        selectionxy.event(bounds=(0, 0, 4, 2))
        current_obj = linked[()]
        self.check_base_points_like(current_obj.Points.I, lnk_sel, self.data.iloc[:2])
        self.check_overlay_points_like(current_obj.Points.II, lnk_sel, self.data.iloc[[0]])
        buffer.send(self.data.iloc[[2]])
        current_obj = linked[()]
        self.check_base_points_like(current_obj.Points.I, lnk_sel, self.data)
        self.check_overlay_points_like(current_obj.Points.II, lnk_sel, self.data.iloc[[0, 2]])

    def do_crossfilter_points_histogram(self, selection_mode, cross_filter_mode, selected1, selected2, selected3, selected4, points_region1, points_region2, points_region3, points_region4, hist_region2, hist_region3, hist_region4, show_regions=True, dynamic=False):
        points = Points(self.data)
        hist = points.hist('x', adjoin=False, normed=False, num_bins=5)
        if dynamic:
            hist_orig = hist
            points = hv.util.Dynamic(points)
        else:
            hist_orig = hist
        lnk_sel = link_selections.instance(selection_mode=selection_mode, cross_filter_mode=cross_filter_mode, show_regions=show_regions, selected_color='#00ff00', unselected_color='#ff0000')
        linked = lnk_sel(points + hist)
        current_obj = linked[()]
        self.check_base_points_like(current_obj[0][()].Points.I, lnk_sel)
        self.check_overlay_points_like(current_obj[0][()].Points.II, lnk_sel, self.data)
        self.assertEqual(len(current_obj[0][()].Curve.I), 0)
        base_hist = current_obj[1][()].Histogram.I
        self.assertEqual(self.element_color(base_hist), lnk_sel.unselected_color)
        self.assertEqual(base_hist.data, hist_orig.data)
        selection_hist = current_obj[1][()].Histogram.II
        self.assertEqual(self.element_color(selection_hist), self.expected_selection_color(selection_hist, lnk_sel))
        self.assertEqual(selection_hist, base_hist)
        region_hist = current_obj[1][()].NdOverlay.I.last
        self.assertEqual(region_hist.data, [None, None])
        points_selectionxy = TestLinkSelections.get_value_with_key_type(lnk_sel._selection_expr_streams, hv.Points).input_streams[0].input_stream.input_streams[0]
        self.assertIsInstance(points_selectionxy, SelectionXY)
        points_selectionxy.event(bounds=(1, 1, 4, 4))
        current_obj = linked[()]
        self.check_base_points_like(current_obj[0][()].Points.I, lnk_sel)
        self.check_overlay_points_like(current_obj[0][()].Points.II, lnk_sel, self.data.iloc[selected1])
        region_bounds = current_obj[0][()].Rectangles.I
        self.assertEqual(region_bounds, Rectangles(points_region1))
        if show_regions:
            self.assertEqual(self.element_color(region_bounds), box_region_color)
        hist_selectionxy = TestLinkSelections.get_value_with_key_type(lnk_sel._selection_expr_streams, hv.Histogram).input_streams[0].input_stream.input_streams[0]
        self.assertIsInstance(hist_selectionxy, SelectionXY)
        hist_selectionxy.event(bounds=(0, 0, 2.5, 2))
        points_unsel, points_sel, points_region, points_region_poly = current_obj[0][()].values()
        self.check_overlay_points_like(points_sel, lnk_sel, self.data.iloc[selected2])
        self.assertEqual(points_region, Rectangles(points_region2))
        base_hist, region_hist, sel_hist = current_obj[1][()].values()
        self.assertEqual(self.element_color(base_hist), lnk_sel.unselected_color)
        self.assertEqual(base_hist.data, hist_orig.data)
        if show_regions:
            self.assertEqual(self.element_color(region_hist.last), hist_region_color)
        if not len(hist_region2) and lnk_sel.selection_mode != 'inverse':
            self.assertEqual(len(region_hist), 1)
        else:
            self.assertEqual(region_hist.last.data, [0, 2.5])
        self.assertEqual(self.element_color(sel_hist), self.expected_selection_color(sel_hist, lnk_sel))
        self.assertEqual(sel_hist.data, hist_orig.pipeline(hist_orig.dataset.iloc[selected2]).data)
        points_selectionxy = TestLinkSelections.get_value_with_key_type(lnk_sel._selection_expr_streams, hv.Points).input_streams[0].input_stream.input_streams[0]
        self.assertIsInstance(points_selectionxy, SelectionXY)
        points_selectionxy.event(bounds=(0, 0, 4, 2.5))
        self.check_overlay_points_like(current_obj[0][()].Points.II, lnk_sel, self.data.iloc[selected3])
        region_bounds = current_obj[0][()].Rectangles.I
        self.assertEqual(region_bounds, Rectangles(points_region3))
        selection_hist = current_obj[1][()].Histogram.II
        self.assertEqual(selection_hist.data, hist_orig.pipeline(hist_orig.dataset.iloc[selected3]).data)
        region_hist = current_obj[1][()].NdOverlay.I.last
        if not len(hist_region3) and lnk_sel.selection_mode != 'inverse':
            self.assertEqual(len(region_hist), 1)
        else:
            self.assertEqual(region_hist.data, [0, 2.5])
        hist_selectionxy = TestLinkSelections.get_value_with_key_type(lnk_sel._selection_expr_streams, hv.Histogram).input_streams[0].input_stream.input_streams[0]
        self.assertIsInstance(hist_selectionxy, SelectionXY)
        hist_selectionxy.event(bounds=(1.5, 0, 3.5, 2))
        self.check_overlay_points_like(current_obj[0][()].Points.II, lnk_sel, self.data.iloc[selected4])
        region_bounds = current_obj[0][()].Rectangles.I
        self.assertEqual(region_bounds, Rectangles(points_region4))
        region_hist = current_obj[1][()].NdOverlay.I.last
        if show_regions:
            self.assertEqual(self.element_color(region_hist), hist_region_color)
        if not len(hist_region4) and lnk_sel.selection_mode != 'inverse':
            self.assertEqual(len(region_hist), 1)
        elif lnk_sel.cross_filter_mode != 'overwrite' and (not (lnk_sel.cross_filter_mode == 'intersect' and lnk_sel.selection_mode == 'overwrite')):
            self.assertEqual(region_hist.data, [0, 3.5])
        else:
            self.assertEqual(region_hist.data, [1.5, 3.5])
        selection_hist = current_obj[1][()].Histogram.II
        self.assertEqual(self.element_color(selection_hist), self.expected_selection_color(selection_hist, lnk_sel))
        self.assertEqual(selection_hist.data, hist_orig.pipeline(hist_orig.dataset.iloc[selected4]).data)

    def test_points_histogram_overwrite_overwrite(self, dynamic=False):
        self.do_crossfilter_points_histogram(selection_mode='overwrite', cross_filter_mode='overwrite', selected1=[1, 2], selected2=[0, 1], selected3=[0, 2], selected4=[1, 2], points_region1=[(1, 1, 4, 4)], points_region2=[], points_region3=[(0, 0, 4, 2.5)], points_region4=[], hist_region2=[0, 1], hist_region3=[], hist_region4=[1, 2], dynamic=dynamic)

    def test_points_histogram_overwrite_overwrite_dynamic(self):
        self.test_points_histogram_overwrite_overwrite(dynamic=True)

    def test_points_histogram_intersect_overwrite(self, dynamic=False):
        self.do_crossfilter_points_histogram(selection_mode='intersect', cross_filter_mode='overwrite', selected1=[1, 2], selected2=[0, 1], selected3=[0, 2], selected4=[1, 2], points_region1=[(1, 1, 4, 4)], points_region2=[], points_region3=[(0, 0, 4, 2.5)], points_region4=[], hist_region2=[0, 1], hist_region3=[], hist_region4=[1, 2], dynamic=dynamic)

    def test_points_histogram_intersect_overwrite_dynamic(self):
        self.test_points_histogram_intersect_overwrite(dynamic=True)

    def test_points_histogram_union_overwrite(self, dynamic=False):
        self.do_crossfilter_points_histogram(selection_mode='union', cross_filter_mode='overwrite', selected1=[1, 2], selected2=[0, 1], selected3=[0, 2], selected4=[1, 2], points_region1=[(1, 1, 4, 4)], points_region2=[], points_region3=[(0, 0, 4, 2.5)], points_region4=[], hist_region2=[0, 1], hist_region3=[], hist_region4=[1, 2], dynamic=dynamic)

    def test_points_histogram_union_overwrite_dynamic(self):
        self.test_points_histogram_union_overwrite(dynamic=True)

    def test_points_histogram_overwrite_intersect(self, dynamic=False):
        self.do_crossfilter_points_histogram(selection_mode='overwrite', cross_filter_mode='intersect', selected1=[1, 2], selected2=[1], selected3=[0], selected4=[2], points_region1=[(1, 1, 4, 4)], points_region2=[(1, 1, 4, 4)], points_region3=[(0, 0, 4, 2.5)], points_region4=[(0, 0, 4, 2.5)], hist_region2=[0, 1], hist_region3=[0, 1], hist_region4=[1, 2], dynamic=dynamic)

    def test_points_histogram_overwrite_intersect_dynamic(self):
        self.test_points_histogram_overwrite_intersect(dynamic=True)

    def test_points_histogram_overwrite_intersect_hide_region(self, dynamic=False):
        self.do_crossfilter_points_histogram(selection_mode='overwrite', cross_filter_mode='intersect', selected1=[1, 2], selected2=[1], selected3=[0], selected4=[2], points_region1=[], points_region2=[], points_region3=[], points_region4=[], hist_region2=[], hist_region3=[], hist_region4=[], show_regions=False, dynamic=dynamic)

    def test_points_histogram_overwrite_intersect_hide_region_dynamic(self):
        self.test_points_histogram_overwrite_intersect_hide_region(dynamic=True)

    def test_points_histogram_intersect_intersect(self, dynamic=False):
        self.do_crossfilter_points_histogram(selection_mode='intersect', cross_filter_mode='intersect', selected1=[1, 2], selected2=[1], selected3=[], selected4=[], points_region1=[(1, 1, 4, 4)], points_region2=[(1, 1, 4, 4)], points_region3=[(1, 1, 4, 4), (0, 0, 4, 2.5)], points_region4=[(1, 1, 4, 4), (0, 0, 4, 2.5)], hist_region2=[0, 1], hist_region3=[0, 1], hist_region4=[1], dynamic=dynamic)

    def test_points_histogram_intersect_intersect_dynamic(self):
        self.test_points_histogram_intersect_intersect(dynamic=True)

    def test_points_histogram_union_intersect(self, dynamic=False):
        self.do_crossfilter_points_histogram(selection_mode='union', cross_filter_mode='intersect', selected1=[1, 2], selected2=[1], selected3=[0, 1], selected4=[0, 1, 2], points_region1=[(1, 1, 4, 4)], points_region2=[(1, 1, 4, 4)], points_region3=[(1, 1, 4, 4), (0, 0, 4, 2.5)], points_region4=[(1, 1, 4, 4), (0, 0, 4, 2.5)], hist_region2=[0, 1], hist_region3=[0, 1], hist_region4=[0, 1, 2], dynamic=dynamic)

    def test_points_histogram_union_intersect_dynamic(self):
        self.test_points_histogram_union_intersect(dynamic=True)

    def test_points_histogram_inverse_intersect(self, dynamic=False):
        self.do_crossfilter_points_histogram(selection_mode='inverse', cross_filter_mode='intersect', selected1=[0], selected2=[], selected3=[], selected4=[], points_region1=[(1, 1, 4, 4)], points_region2=[(1, 1, 4, 4)], points_region3=[(1, 1, 4, 4), (0, 0, 4, 2.5)], points_region4=[(1, 1, 4, 4), (0, 0, 4, 2.5)], hist_region2=[], hist_region3=[], hist_region4=[], dynamic=dynamic)

    def test_points_histogram_inverse_intersect_dynamic(self):
        self.test_points_histogram_inverse_intersect(dynamic=True)