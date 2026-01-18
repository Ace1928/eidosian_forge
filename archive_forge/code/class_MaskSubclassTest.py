from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
class MaskSubclassTest(unittest.TestCase):
    """Test subclassed Masks."""

    def test_subclass_mask(self):
        """Ensures the Mask class can be subclassed."""
        mask = SubMask((5, 3), fill=True)
        self.assertIsInstance(mask, pygame.mask.Mask)
        self.assertIsInstance(mask, SubMask)
        self.assertTrue(mask.test_attribute)

    def test_subclass_copy(self):
        """Ensures copy works for subclassed Masks."""
        mask = SubMask((65, 2), fill=True)
        for mask_copy in (mask.copy(), copy.copy(mask)):
            self.assertIsInstance(mask_copy, pygame.mask.Mask)
            self.assertIsInstance(mask_copy, SubMask)
            self.assertIsNot(mask_copy, mask)
            assertMaskEqual(self, mask_copy, mask)
            self.assertFalse(hasattr(mask_copy, 'test_attribute'))

    def test_subclass_copy__override_copy(self):
        """Ensures copy works for subclassed Masks overriding copy."""
        mask = SubMaskCopy((65, 2), fill=True)
        for i, mask_copy in enumerate((mask.copy(), copy.copy(mask))):
            self.assertIsInstance(mask_copy, pygame.mask.Mask)
            self.assertIsInstance(mask_copy, SubMaskCopy)
            self.assertIsNot(mask_copy, mask)
            assertMaskEqual(self, mask_copy, mask)
            if 1 == i:
                self.assertFalse(hasattr(mask_copy, 'test_attribute'))
            else:
                self.assertTrue(mask_copy.test_attribute)

    def test_subclass_copy__override_dunder_copy(self):
        """Ensures copy works for subclassed Masks overriding __copy__."""
        mask = SubMaskDunderCopy((65, 2), fill=True)
        for mask_copy in (mask.copy(), copy.copy(mask)):
            self.assertIsInstance(mask_copy, pygame.mask.Mask)
            self.assertIsInstance(mask_copy, SubMaskDunderCopy)
            self.assertIsNot(mask_copy, mask)
            assertMaskEqual(self, mask_copy, mask)
            self.assertTrue(mask_copy.test_attribute)

    def test_subclass_copy__override_both_copy_methods(self):
        """Ensures copy works for subclassed Masks overriding copy/__copy__."""
        mask = SubMaskCopyAndDunderCopy((65, 2), fill=True)
        for mask_copy in (mask.copy(), copy.copy(mask)):
            self.assertIsInstance(mask_copy, pygame.mask.Mask)
            self.assertIsInstance(mask_copy, SubMaskCopyAndDunderCopy)
            self.assertIsNot(mask_copy, mask)
            assertMaskEqual(self, mask_copy, mask)
            self.assertTrue(mask_copy.test_attribute)

    def test_subclass_get_size(self):
        """Ensures get_size works for subclassed Masks."""
        expected_size = (2, 3)
        mask = SubMask(expected_size)
        size = mask.get_size()
        self.assertEqual(size, expected_size)

    def test_subclass_mask_get_rect(self):
        """Ensures get_rect works for subclassed Masks."""
        expected_rect = pygame.Rect((0, 0), (65, 33))
        mask = SubMask(expected_rect.size, fill=True)
        rect = mask.get_rect()
        self.assertEqual(rect, expected_rect)

    def test_subclass_get_at(self):
        """Ensures get_at works for subclassed Masks."""
        expected_bit = 1
        mask = SubMask((3, 2), fill=True)
        bit = mask.get_at((0, 0))
        self.assertEqual(bit, expected_bit)

    def test_subclass_set_at(self):
        """Ensures set_at works for subclassed Masks."""
        expected_bit = 1
        expected_count = 1
        pos = (0, 0)
        mask = SubMask(fill=False, size=(4, 2))
        mask.set_at(pos)
        self.assertEqual(mask.get_at(pos), expected_bit)
        self.assertEqual(mask.count(), expected_count)

    def test_subclass_overlap(self):
        """Ensures overlap works for subclassed Masks."""
        expected_pos = (0, 0)
        mask_size = (2, 3)
        masks = (pygame.mask.Mask(fill=True, size=mask_size), SubMask(mask_size, True))
        arg_masks = (pygame.mask.Mask(fill=True, size=mask_size), SubMask(mask_size, True))
        for mask in masks:
            for arg_mask in arg_masks:
                overlap_pos = mask.overlap(arg_mask, (0, 0))
                self.assertEqual(overlap_pos, expected_pos)

    def test_subclass_overlap_area(self):
        """Ensures overlap_area works for subclassed Masks."""
        mask_size = (3, 2)
        expected_count = mask_size[0] * mask_size[1]
        masks = (pygame.mask.Mask(fill=True, size=mask_size), SubMask(mask_size, True))
        arg_masks = (pygame.mask.Mask(fill=True, size=mask_size), SubMask(mask_size, True))
        for mask in masks:
            for arg_mask in arg_masks:
                overlap_count = mask.overlap_area(arg_mask, (0, 0))
                self.assertEqual(overlap_count, expected_count)

    def test_subclass_overlap_mask(self):
        """Ensures overlap_mask works for subclassed Masks."""
        expected_size = (4, 5)
        expected_count = expected_size[0] * expected_size[1]
        masks = (pygame.mask.Mask(fill=True, size=expected_size), SubMask(expected_size, True))
        arg_masks = (pygame.mask.Mask(fill=True, size=expected_size), SubMask(expected_size, True))
        for mask in masks:
            for arg_mask in arg_masks:
                overlap_mask = mask.overlap_mask(arg_mask, (0, 0))
                self.assertIsInstance(overlap_mask, pygame.mask.Mask)
                self.assertNotIsInstance(overlap_mask, SubMask)
                self.assertEqual(overlap_mask.count(), expected_count)
                self.assertEqual(overlap_mask.get_size(), expected_size)

    def test_subclass_fill(self):
        """Ensures fill works for subclassed Masks."""
        mask_size = (2, 4)
        expected_count = mask_size[0] * mask_size[1]
        mask = SubMask(fill=False, size=mask_size)
        mask.fill()
        self.assertEqual(mask.count(), expected_count)

    def test_subclass_clear(self):
        """Ensures clear works for subclassed Masks."""
        mask_size = (4, 3)
        expected_count = 0
        mask = SubMask(mask_size, True)
        mask.clear()
        self.assertEqual(mask.count(), expected_count)

    def test_subclass_invert(self):
        """Ensures invert works for subclassed Masks."""
        mask_size = (1, 4)
        expected_count = mask_size[0] * mask_size[1]
        mask = SubMask(fill=False, size=mask_size)
        mask.invert()
        self.assertEqual(mask.count(), expected_count)

    def test_subclass_scale(self):
        """Ensures scale works for subclassed Masks."""
        expected_size = (5, 2)
        mask = SubMask((1, 4))
        scaled_mask = mask.scale(expected_size)
        self.assertIsInstance(scaled_mask, pygame.mask.Mask)
        self.assertNotIsInstance(scaled_mask, SubMask)
        self.assertEqual(scaled_mask.get_size(), expected_size)

    def test_subclass_draw(self):
        """Ensures draw works for subclassed Masks."""
        mask_size = (5, 4)
        expected_count = mask_size[0] * mask_size[1]
        arg_masks = (pygame.mask.Mask(fill=True, size=mask_size), SubMask(mask_size, True))
        for mask in (pygame.mask.Mask(mask_size), SubMask(mask_size)):
            for arg_mask in arg_masks:
                mask.clear()
                mask.draw(arg_mask, (0, 0))
                self.assertEqual(mask.count(), expected_count)

    def test_subclass_erase(self):
        """Ensures erase works for subclassed Masks."""
        mask_size = (3, 4)
        expected_count = 0
        masks = (pygame.mask.Mask(mask_size, True), SubMask(mask_size, True))
        arg_masks = (pygame.mask.Mask(mask_size, True), SubMask(mask_size, True))
        for mask in masks:
            for arg_mask in arg_masks:
                mask.fill()
                mask.erase(arg_mask, (0, 0))
                self.assertEqual(mask.count(), expected_count)

    def test_subclass_count(self):
        """Ensures count works for subclassed Masks."""
        mask_size = (5, 2)
        expected_count = mask_size[0] * mask_size[1] - 1
        mask = SubMask(fill=True, size=mask_size)
        mask.set_at((1, 1), 0)
        count = mask.count()
        self.assertEqual(count, expected_count)

    def test_subclass_centroid(self):
        """Ensures centroid works for subclassed Masks."""
        expected_centroid = (0, 0)
        mask_size = (3, 2)
        mask = SubMask((3, 2))
        centroid = mask.centroid()
        self.assertEqual(centroid, expected_centroid)

    def test_subclass_angle(self):
        """Ensures angle works for subclassed Masks."""
        expected_angle = 0.0
        mask = SubMask(size=(5, 4))
        angle = mask.angle()
        self.assertAlmostEqual(angle, expected_angle)

    def test_subclass_outline(self):
        """Ensures outline works for subclassed Masks."""
        expected_outline = []
        mask = SubMask((3, 4))
        outline = mask.outline()
        self.assertListEqual(outline, expected_outline)

    def test_subclass_convolve(self):
        """Ensures convolve works for subclassed Masks."""
        width, height = (7, 5)
        mask_size = (width, height)
        expected_count = 0
        expected_size = (max(0, width * 2 - 1), max(0, height * 2 - 1))
        arg_masks = (pygame.mask.Mask(mask_size), SubMask(mask_size))
        output_masks = (pygame.mask.Mask(mask_size), SubMask(mask_size))
        for mask in (pygame.mask.Mask(mask_size), SubMask(mask_size)):
            for arg_mask in arg_masks:
                convolve_mask = mask.convolve(arg_mask)
                self.assertIsInstance(convolve_mask, pygame.mask.Mask)
                self.assertNotIsInstance(convolve_mask, SubMask)
                self.assertEqual(convolve_mask.count(), expected_count)
                self.assertEqual(convolve_mask.get_size(), expected_size)
                for output_mask in output_masks:
                    convolve_mask = mask.convolve(arg_mask, output_mask)
                    self.assertIsInstance(convolve_mask, pygame.mask.Mask)
                    self.assertEqual(convolve_mask.count(), expected_count)
                    self.assertEqual(convolve_mask.get_size(), mask_size)
                    if isinstance(output_mask, SubMask):
                        self.assertIsInstance(convolve_mask, SubMask)
                    else:
                        self.assertNotIsInstance(convolve_mask, SubMask)

    def test_subclass_connected_component(self):
        """Ensures connected_component works for subclassed Masks."""
        expected_count = 0
        expected_size = (3, 4)
        mask = SubMask(expected_size)
        cc_mask = mask.connected_component()
        self.assertIsInstance(cc_mask, pygame.mask.Mask)
        self.assertNotIsInstance(cc_mask, SubMask)
        self.assertEqual(cc_mask.count(), expected_count)
        self.assertEqual(cc_mask.get_size(), expected_size)

    def test_subclass_connected_components(self):
        """Ensures connected_components works for subclassed Masks."""
        expected_ccs = []
        mask = SubMask((5, 4))
        ccs = mask.connected_components()
        self.assertListEqual(ccs, expected_ccs)

    def test_subclass_get_bounding_rects(self):
        """Ensures get_bounding_rects works for subclassed Masks."""
        expected_bounding_rects = []
        mask = SubMask((3, 2))
        bounding_rects = mask.get_bounding_rects()
        self.assertListEqual(bounding_rects, expected_bounding_rects)

    def test_subclass_to_surface(self):
        """Ensures to_surface works for subclassed Masks."""
        expected_color = pygame.Color('blue')
        size = (5, 3)
        mask = SubMask(size, fill=True)
        surface = pygame.Surface(size, SRCALPHA, 32)
        surface.fill(pygame.Color('red'))
        to_surface = mask.to_surface(surface, setcolor=expected_color)
        self.assertIs(to_surface, surface)
        self.assertEqual(to_surface.get_size(), size)
        assertSurfaceFilled(self, to_surface, expected_color)