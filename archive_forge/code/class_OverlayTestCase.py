from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
class OverlayTestCase(ElementTestCase):
    """
    The tests here match those in LayoutTestCase; Overlays inherit
    from Layout and behave in a very similar way (except for being
    associated with * instead of the + operator)
    """

    def test_overlay_keys(self):
        t = self.el1 * self.el2
        self.assertEqual(t.keys(), [('Element', 'I'), ('Element', 'II')])

    def test_overlay_keys_2(self):
        t = Overlay([self.el1, self.el2])
        self.assertEqual(t.keys(), [('Element', 'I'), ('Element', 'II')])

    def test_overlay_values(self):
        t = self.el1 * self.el2
        self.assertEqual(t.values(), [self.el1, self.el2])

    def test_overlay_values_2(self):
        t = Overlay([self.el1, self.el2])
        self.assertEqual(t.values(), [self.el1, self.el2])

    def test_triple_overlay_keys(self):
        t = self.el1 * self.el2 * self.el3
        expected_keys = [('Element', 'I'), ('Element', 'II'), ('Element', 'III')]
        self.assertEqual(t.keys(), expected_keys)

    def test_triple_overlay_values(self):
        t = self.el1 * self.el2 * self.el3
        self.assertEqual(t.values(), [self.el1, self.el2, self.el3])

    def test_overlay_varying_value_keys(self):
        t = self.el1 * self.el4
        self.assertEqual(t.keys(), [('Element', 'I'), ('ValA', 'I')])

    def test_overlay_varying_value_keys2(self):
        t = self.el4 * self.el5
        self.assertEqual(t.keys(), [('ValA', 'I'), ('ValB', 'I')])

    def test_triple_overlay_varying_value_keys(self):
        t = self.el1 * self.el4 * self.el2 * self.el3
        expected_keys = [('Element', 'I'), ('ValA', 'I'), ('Element', 'II'), ('Element', 'III')]
        self.assertEqual(t.keys(), expected_keys)

    def test_four_overlay_varying_value_values(self):
        t = self.el1 * self.el4 * self.el2 * self.el3
        self.assertEqual(t.values(), [self.el1, self.el4, self.el2, self.el3])

    def test_overlay_varying_label_keys(self):
        t = self.el1 * self.el6
        self.assertEqual(t.keys(), [('Element', 'I'), ('Element', 'LabelA')])

    def test_triple_overlay_varying_label_keys(self):
        t = self.el1 * self.el6 * self.el2
        expected_keys = [('Element', 'I'), ('Element', 'LabelA'), ('Element', 'II')]
        self.assertEqual(t.keys(), expected_keys)

    def test_overlay_varying_label_keys2(self):
        t = self.el7 * self.el8
        self.assertEqual(t.keys(), [('ValA', 'LabelA'), ('ValA', 'LabelB')])

    def test_overlay_varying_label_and_values_keys(self):
        t = self.el6 * self.el7 * self.el8
        expected_keys = [('Element', 'LabelA'), ('ValA', 'LabelA'), ('ValA', 'LabelB')]
        self.assertEqual(t.keys(), expected_keys)

    def test_overlay_varying_label_and_values_values(self):
        t = self.el6 * self.el7 * self.el8
        self.assertEqual(t.values(), [self.el6, self.el7, self.el8])

    def test_deep_overlay_keys(self):
        o1 = self.el1 * self.el2
        o2 = self.el1 * self.el2
        o3 = self.el1 * self.el2
        t = o1 * o2 * o3
        expected_keys = [('Element', 'I'), ('Element', 'II'), ('Element', 'III'), ('Element', 'IV'), ('Element', 'V'), ('Element', 'VI')]
        self.assertEqual(t.keys(), expected_keys)

    def test_deep_overlay_values(self):
        o1 = self.el1 * self.el2
        o2 = self.el1 * self.el2
        o3 = self.el1 * self.el2
        t = o1 * o2 * o3
        self.assertEqual(t.values(), [self.el1, self.el2, self.el1, self.el2, self.el1, self.el2])

    def test_overlay_associativity(self):
        o1 = self.el1 * self.el2 * self.el3
        o2 = self.el1 * self.el2 * self.el3
        o3 = self.el1 * (self.el2 * self.el3)
        self.assertEqual(o1.keys(), o2.keys())
        self.assertEqual(o2.keys(), o3.keys())

    def test_overlay_constructor1(self):
        t = Overlay([self.el1])
        self.assertEqual(t.keys(), [('Element', 'I')])

    def test_overlay_constructor2(self):
        t = Overlay([self.el8])
        self.assertEqual(t.keys(), [('ValA', 'LabelB')])

    def test_overlay_group(self):
        t1 = self.el1 * self.el2
        t2 = Overlay(list(t1.relabel(group='NewValue', depth=1)))
        self.assertEqual(t2.keys(), [('NewValue', 'I'), ('NewValue', 'II')])

    def test_overlay_quadruple_1(self):
        t = self.el1 * self.el1 * self.el1 * self.el1
        self.assertEqual(t.keys(), [('Element', 'I'), ('Element', 'II'), ('Element', 'III'), ('Element', 'IV')])

    def test_overlay_quadruple_2(self):
        t = self.el6 * self.el6 * self.el6 * self.el6
        self.assertEqual(t.keys(), [('Element', 'LabelA', 'I'), ('Element', 'LabelA', 'II'), ('Element', 'LabelA', 'III'), ('Element', 'LabelA', 'IV')])

    def test_overlay_constructor_with_layouts(self):
        layout1 = self.el1 + self.el4
        layout2 = self.el2 + self.el5
        paths = Layout([layout1, layout2]).keys()
        self.assertEqual(paths, [('Element', 'I'), ('ValA', 'I'), ('Element', 'II'), ('ValB', 'I')])

    def test_overlay_constructor_with_mixed_types(self):
        overlay1 = self.el1 + self.el4 + self.el7
        overlay2 = self.el2 + self.el5 + self.el8
        paths = Layout([overlay1, overlay2, self.el3]).keys()
        self.assertEqual(paths, [('Element', 'I'), ('ValA', 'I'), ('ValA', 'LabelA'), ('Element', 'II'), ('ValB', 'I'), ('ValA', 'LabelB'), ('Element', 'III')])

    def test_overlay_constructor_retains_custom_path(self):
        overlay = Overlay([('Custom', self.el1)])
        paths = Overlay([overlay, self.el2]).keys()
        self.assertEqual(paths, [('Custom', 'I'), ('Element', 'I')])

    def test_overlay_constructor_retains_custom_path_with_label(self):
        overlay = Overlay([('Custom', self.el6)])
        paths = Overlay([overlay, self.el2]).keys()
        self.assertEqual(paths, [('Custom', 'LabelA'), ('Element', 'I')])

    def test_overlay_with_holomap(self):
        overlay = Overlay([('Custom', self.el6)])
        composite = overlay * HoloMap({0: Element(None, group='HoloMap')})
        self.assertEqual(composite.last.keys(), [('Custom', 'LabelA'), ('HoloMap', 'I')])

    def test_overlay_id_inheritance(self):
        overlay = Overlay([], id=1)
        self.assertEqual(overlay.clone().id, 1)
        self.assertEqual(overlay.clone()._plot_id, overlay._plot_id)
        self.assertNotEqual(overlay.clone([])._plot_id, overlay._plot_id)