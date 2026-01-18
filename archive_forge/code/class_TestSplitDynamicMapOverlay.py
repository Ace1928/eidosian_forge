import numpy as np
from holoviews import Dimension, NdOverlay, Overlay
from holoviews.core.options import Cycle, Store
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import operation
from holoviews.plotting.util import (
from holoviews.streams import PointerX
class TestSplitDynamicMapOverlay(ComparisonTestCase):
    """
    Tests the split_dmap_overlay utility
    """

    def setUp(self):
        self.dmap_element = DynamicMap(lambda: Image([]))
        self.dmap_overlay = DynamicMap(lambda: Overlay([Curve([]), Points([])]))
        self.dmap_ndoverlay = DynamicMap(lambda: NdOverlay({0: Curve([]), 1: Curve([])}))
        self.element = Scatter([])
        self.el1, self.el2 = (Path([]), HLine(0))
        self.overlay = Overlay([self.el1, self.el2])
        self.ndoverlay = NdOverlay({0: VectorField([]), 1: VectorField([])})

    def test_dmap_ndoverlay(self):
        test = self.dmap_ndoverlay
        initialize_dynamic(test)
        layers = [self.dmap_ndoverlay, self.dmap_ndoverlay]
        self.assertEqual(split_dmap_overlay(test), layers)

    def test_dmap_overlay(self):
        test = self.dmap_overlay
        initialize_dynamic(test)
        layers = [self.dmap_overlay, self.dmap_overlay]
        self.assertEqual(split_dmap_overlay(test), layers)

    def test_dmap_element_mul_dmap_overlay(self):
        test = self.dmap_element * self.dmap_overlay
        initialize_dynamic(test)
        layers = [self.dmap_element, self.dmap_overlay, self.dmap_overlay]
        self.assertEqual(split_dmap_overlay(test), layers)

    def test_dmap_element_mul_dmap_ndoverlay(self):
        test = self.dmap_element * self.dmap_ndoverlay
        initialize_dynamic(test)
        layers = [self.dmap_element, self.dmap_ndoverlay]
        self.assertEqual(split_dmap_overlay(test), layers)

    def test_dmap_element_mul_element(self):
        test = self.dmap_element * self.element
        initialize_dynamic(test)
        layers = [self.dmap_element, self.element]
        self.assertEqual(split_dmap_overlay(test), layers)

    def test_dmap_element_mul_overlay(self):
        test = self.dmap_element * self.overlay
        initialize_dynamic(test)
        layers = [self.dmap_element, self.el1, self.el2]
        self.assertEqual(split_dmap_overlay(test), layers)

    def test_dmap_element_mul_ndoverlay(self):
        test = self.dmap_element * self.ndoverlay
        initialize_dynamic(test)
        layers = [self.dmap_element, self.ndoverlay]
        self.assertEqual(split_dmap_overlay(test), layers)

    def test_dmap_overlay_mul_dmap_ndoverlay(self):
        test = self.dmap_overlay * self.dmap_ndoverlay
        initialize_dynamic(test)
        layers = [self.dmap_overlay, self.dmap_overlay, self.dmap_ndoverlay]
        self.assertEqual(split_dmap_overlay(test), layers)

    def test_dmap_overlay_mul_element(self):
        test = self.dmap_overlay * self.element
        initialize_dynamic(test)
        layers = [self.dmap_overlay, self.dmap_overlay, self.element]
        self.assertEqual(split_dmap_overlay(test), layers)

    def test_dmap_overlay_mul_overlay(self):
        test = self.dmap_overlay * self.overlay
        initialize_dynamic(test)
        layers = [self.dmap_overlay, self.dmap_overlay, self.el1, self.el2]
        self.assertEqual(split_dmap_overlay(test), layers)

    def test_dmap_all_combinations(self):
        test = self.dmap_overlay * self.element * self.dmap_ndoverlay * self.overlay * self.dmap_element * self.ndoverlay
        initialize_dynamic(test)
        layers = [self.dmap_overlay, self.dmap_overlay, self.element, self.dmap_ndoverlay, self.el1, self.el2, self.dmap_element, self.ndoverlay]
        self.assertEqual(split_dmap_overlay(test), layers)

    def test_dmap_overlay_operation_mul_dmap_ndoverlay(self):
        mapped = operation(self.dmap_overlay)
        test = mapped * self.dmap_ndoverlay
        initialize_dynamic(test)
        layers = [mapped, mapped, self.dmap_ndoverlay]
        self.assertEqual(split_dmap_overlay(test), layers)

    def test_dmap_overlay_linked_operation_mul_dmap_ndoverlay(self):
        mapped = operation(self.dmap_overlay, link_inputs=True)
        test = mapped * self.dmap_ndoverlay
        initialize_dynamic(test)
        layers = [mapped, mapped, self.dmap_ndoverlay]
        self.assertEqual(split_dmap_overlay(test), layers)

    def test_dmap_overlay_linked_operation_mul_dmap_element_ndoverlay(self):
        mapped = self.dmap_overlay.map(lambda x: x.get(0), Overlay)
        test = mapped * self.element * self.dmap_ndoverlay
        initialize_dynamic(test)
        layers = [mapped, self.element, self.dmap_ndoverlay]
        self.assertEqual(split_dmap_overlay(test), layers)