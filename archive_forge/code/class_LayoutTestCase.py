from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
class LayoutTestCase(ElementTestCase):

    def setUp(self):
        super().setUp()

    def test_layouttree_keys_1(self):
        t = self.el1 + self.el2
        self.assertEqual(t.keys(), [('Element', 'I'), ('Element', 'II')])

    def test_layouttree_keys_2(self):
        t = Layout([self.el1, self.el2])
        self.assertEqual(t.keys(), [('Element', 'I'), ('Element', 'II')])

    def test_layouttree_deduplicate(self):
        for i in range(2, 10):
            l = Layout([Element([], label='0') for _ in range(i)])
            self.assertEqual(len(l), i)

    def test_layouttree_values_1(self):
        t = self.el1 + self.el2
        self.assertEqual(t.values(), [self.el1, self.el2])

    def test_layouttree_values_2(self):
        t = Layout([self.el1, self.el2])
        self.assertEqual(t.values(), [self.el1, self.el2])

    def test_triple_layouttree_keys(self):
        t = self.el1 + self.el2 + self.el3
        expected_keys = [('Element', 'I'), ('Element', 'II'), ('Element', 'III')]
        self.assertEqual(t.keys(), expected_keys)

    def test_triple_layouttree_values(self):
        t = self.el1 + self.el2 + self.el3
        self.assertEqual(t.values(), [self.el1, self.el2, self.el3])

    def test_layouttree_varying_value_keys(self):
        t = self.el1 + self.el4
        self.assertEqual(t.keys(), [('Element', 'I'), ('ValA', 'I')])

    def test_layouttree_varying_value_keys2(self):
        t = self.el4 + self.el5
        self.assertEqual(t.keys(), [('ValA', 'I'), ('ValB', 'I')])

    def test_triple_layouttree_varying_value_keys(self):
        t = self.el1 + self.el4 + self.el2 + self.el3
        expected_keys = [('Element', 'I'), ('ValA', 'I'), ('Element', 'II'), ('Element', 'III')]
        self.assertEqual(t.keys(), expected_keys)

    def test_four_layouttree_varying_value_values(self):
        t = self.el1 + self.el4 + self.el2 + self.el3
        self.assertEqual(t.values(), [self.el1, self.el4, self.el2, self.el3])

    def test_layouttree_varying_label_keys(self):
        t = self.el1 + self.el6
        self.assertEqual(t.keys(), [('Element', 'I'), ('Element', 'LabelA')])

    def test_triple_layouttree_varying_label_keys(self):
        t = self.el1 + self.el6 + self.el2
        expected_keys = [('Element', 'I'), ('Element', 'LabelA'), ('Element', 'II')]
        self.assertEqual(t.keys(), expected_keys)

    def test_layouttree_varying_label_keys2(self):
        t = self.el7 + self.el8
        self.assertEqual(t.keys(), [('ValA', 'LabelA'), ('ValA', 'LabelB')])

    def test_layouttree_varying_label_and_values_keys(self):
        t = self.el6 + self.el7 + self.el8
        expected_keys = [('Element', 'LabelA'), ('ValA', 'LabelA'), ('ValA', 'LabelB')]
        self.assertEqual(t.keys(), expected_keys)

    def test_layouttree_varying_label_and_values_values(self):
        t = self.el6 + self.el7 + self.el8
        self.assertEqual(t.values(), [self.el6, self.el7, self.el8])

    def test_layouttree_associativity(self):
        t1 = self.el1 + self.el2 + self.el3
        t2 = self.el1 + self.el2 + self.el3
        t3 = self.el1 + (self.el2 + self.el3)
        self.assertEqual(t1.keys(), t2.keys())
        self.assertEqual(t2.keys(), t3.keys())

    def test_layouttree_constructor1(self):
        t = Layout([self.el1])
        self.assertEqual(t.keys(), [('Element', 'I')])

    def test_layouttree_constructor2(self):
        t = Layout([self.el8])
        self.assertEqual(t.keys(), [('ValA', 'LabelB')])

    def test_layouttree_group(self):
        t1 = self.el1 + self.el2
        t2 = Layout(list(t1.relabel(group='NewValue')))
        self.assertEqual(t2.keys(), [('NewValue', 'I'), ('NewValue', 'II')])

    def test_layouttree_quadruple_1(self):
        t = self.el1 + self.el1 + self.el1 + self.el1
        self.assertEqual(t.keys(), [('Element', 'I'), ('Element', 'II'), ('Element', 'III'), ('Element', 'IV')])

    def test_layouttree_quadruple_2(self):
        t = self.el6 + self.el6 + self.el6 + self.el6
        self.assertEqual(t.keys(), [('Element', 'LabelA', 'I'), ('Element', 'LabelA', 'II'), ('Element', 'LabelA', 'III'), ('Element', 'LabelA', 'IV')])

    def test_layout_constructor_with_layouts(self):
        layout1 = self.el1 + self.el4
        layout2 = self.el2 + self.el5
        paths = Layout([layout1, layout2]).keys()
        self.assertEqual(paths, [('Element', 'I'), ('ValA', 'I'), ('Element', 'II'), ('ValB', 'I')])

    def test_layout_constructor_with_mixed_types(self):
        layout1 = self.el1 + self.el4 + self.el7
        layout2 = self.el2 + self.el5 + self.el8
        paths = Layout([layout1, layout2, self.el3]).keys()
        self.assertEqual(paths, [('Element', 'I'), ('ValA', 'I'), ('ValA', 'LabelA'), ('Element', 'II'), ('ValB', 'I'), ('ValA', 'LabelB'), ('Element', 'III')])

    def test_layout_constructor_retains_custom_path(self):
        layout = Layout([('Custom', self.el1)])
        paths = Layout([layout, self.el2]).keys()
        self.assertEqual(paths, [('Custom', 'I'), ('Element', 'I')])

    def test_layout_constructor_retains_custom_path_with_label(self):
        layout = Layout([('Custom', self.el6)])
        paths = Layout([layout, self.el2]).keys()
        self.assertEqual(paths, [('Custom', 'LabelA'), ('Element', 'I')])

    def test_layout_integer_index(self):
        t = self.el1 + self.el2
        self.assertEqual(t[0], self.el1)
        self.assertEqual(t[1], self.el2)

    def test_layout_overlay_element(self):
        t = (self.el1 + self.el2) * self.el3
        self.assertEqual(t, Layout([self.el1 * self.el3, self.el2 * self.el3]))

    def test_layout_overlay_element_reverse(self):
        t = self.el3 * (self.el1 + self.el2)
        self.assertEqual(t, Layout([self.el3 * self.el1, self.el3 * self.el2]))

    def test_layout_overlay_overlay(self):
        t = (self.el1 + self.el2) * (self.el3 * self.el4)
        self.assertEqual(t, Layout([self.el1 * self.el3 * self.el4, self.el2 * self.el3 * self.el4]))

    def test_layout_overlay_overlay_reverse(self):
        t = self.el3 * self.el4 * (self.el1 + self.el2)
        self.assertEqual(t, Layout([self.el3 * self.el4 * self.el1, self.el3 * self.el4 * self.el2]))

    def test_layout_overlay_holomap(self):
        t = (self.el1 + self.el2) * HoloMap({0: self.el3})
        self.assertEqual(t, Layout([HoloMap({0: self.el1 * self.el3}), HoloMap({0: self.el2 * self.el3})]))

    def test_layout_overlay_holomap_reverse(self):
        t = HoloMap({0: self.el3}) * (self.el1 + self.el2)
        self.assertEqual(t, Layout([HoloMap({0: self.el3 * self.el1}), HoloMap({0: self.el3 * self.el2})]))