import copy
import numpy as np
from numpy.testing import (assert_allclose, assert_almost_equal,
import pytest
from matplotlib import scale
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.transforms import Affine2D, Bbox, TransformedBbox
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
class TestBasicTransform:

    def setup_method(self):
        self.ta1 = mtransforms.Affine2D(shorthand_name='ta1').rotate(np.pi / 2)
        self.ta2 = mtransforms.Affine2D(shorthand_name='ta2').translate(10, 0)
        self.ta3 = mtransforms.Affine2D(shorthand_name='ta3').scale(1, 2)
        self.tn1 = NonAffineForTest(mtransforms.Affine2D().translate(1, 2), shorthand_name='tn1')
        self.tn2 = NonAffineForTest(mtransforms.Affine2D().translate(1, 2), shorthand_name='tn2')
        self.tn3 = NonAffineForTest(mtransforms.Affine2D().translate(1, 2), shorthand_name='tn3')
        self.stack1 = self.ta1 + (self.tn1 + self.ta2) + self.ta3
        self.stack2 = self.ta1 + self.tn1 + self.ta2 + self.ta3
        self.stack2_subset = self.tn1 + self.ta2 + self.ta3

    def test_transform_depth(self):
        assert self.stack1.depth == 4
        assert self.stack2.depth == 4
        assert self.stack2_subset.depth == 3

    def test_left_to_right_iteration(self):
        stack3 = self.ta1 + (self.tn1 + (self.ta2 + self.tn2)) + self.ta3
        target_transforms = [stack3, self.tn1 + (self.ta2 + self.tn2) + self.ta3, self.ta2 + self.tn2 + self.ta3, self.tn2 + self.ta3, self.ta3]
        r = [rh for _, rh in stack3._iter_break_from_left_to_right()]
        assert len(r) == len(target_transforms)
        for target_stack, stack in zip(target_transforms, r):
            assert target_stack == stack

    def test_transform_shortcuts(self):
        assert self.stack1 - self.stack2_subset == self.ta1
        assert self.stack2 - self.stack2_subset == self.ta1
        assert self.stack2_subset - self.stack2 == self.ta1.inverted()
        assert (self.stack2_subset - self.stack2).depth == 1
        with pytest.raises(ValueError):
            self.stack1 - self.stack2
        aff1 = self.ta1 + (self.ta2 + self.ta3)
        aff2 = self.ta2 + self.ta3
        assert aff1 - aff2 == self.ta1
        assert aff1 - self.ta2 == aff1 + self.ta2.inverted()
        assert self.stack1 - self.ta3 == self.ta1 + (self.tn1 + self.ta2)
        assert self.stack2 - self.ta3 == self.ta1 + self.tn1 + self.ta2
        assert self.ta2 + self.ta3 - self.ta3 + self.ta3 == self.ta2 + self.ta3

    def test_contains_branch(self):
        r1 = self.ta2 + self.ta1
        r2 = self.ta2 + self.ta1
        assert r1 == r2
        assert r1 != self.ta1
        assert r1.contains_branch(r2)
        assert r1.contains_branch(self.ta1)
        assert not r1.contains_branch(self.ta2)
        assert not r1.contains_branch(self.ta2 + self.ta2)
        assert r1 == r2
        assert self.stack1.contains_branch(self.ta3)
        assert self.stack2.contains_branch(self.ta3)
        assert self.stack1.contains_branch(self.stack2_subset)
        assert self.stack2.contains_branch(self.stack2_subset)
        assert not self.stack2_subset.contains_branch(self.stack1)
        assert not self.stack2_subset.contains_branch(self.stack2)
        assert self.stack1.contains_branch(self.ta2 + self.ta3)
        assert self.stack2.contains_branch(self.ta2 + self.ta3)
        assert not self.stack1.contains_branch(self.tn1 + self.ta2)

    def test_affine_simplification(self):
        points = np.array([[0, 0], [10, 20], [np.nan, 1], [-1, 0]], dtype=np.float64)
        na_pts = self.stack1.transform_non_affine(points)
        all_pts = self.stack1.transform(points)
        na_expected = np.array([[1.0, 2.0], [-19.0, 12.0], [np.nan, np.nan], [1.0, 1.0]], dtype=np.float64)
        all_expected = np.array([[11.0, 4.0], [-9.0, 24.0], [np.nan, np.nan], [11.0, 2.0]], dtype=np.float64)
        assert_array_almost_equal(na_pts, na_expected)
        assert_array_almost_equal(all_pts, all_expected)
        assert_array_almost_equal(self.stack1.transform_affine(na_pts), all_expected)
        assert_array_almost_equal(self.stack1.get_affine().transform(na_pts), all_expected)
        expected_result = (self.ta2 + self.ta3).get_matrix()
        result = self.stack1.get_affine().get_matrix()
        assert_array_equal(expected_result, result)
        result = self.stack2.get_affine().get_matrix()
        assert_array_equal(expected_result, result)