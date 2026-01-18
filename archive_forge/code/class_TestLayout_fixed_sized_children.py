import unittest
import pytest
from kivy.tests.common import GraphicUnitTest
from kivy.uix.gridlayout import GridLayout
class TestLayout_fixed_sized_children:

    def compute_layout(self, *, n_cols, n_rows, ori, n_children):
        from kivy.uix.widget import Widget
        from kivy.uix.gridlayout import GridLayout
        gl = GridLayout(cols=n_cols, rows=n_rows, orientation=ori, pos=(0, 0))
        gl.bind(minimum_size=gl.setter('size'))
        for __ in range(n_children):
            gl.add_widget(Widget(size_hint=(None, None), size=(100, 100), pos=(8, 8)))
        gl.do_layout()
        return [tuple(c.pos) for c in reversed(gl.children)]

    @pytest.mark.parametrize('n_cols, n_rows', [(1, None), (None, 1), (1, 1)])
    def test_1x1(self, n_cols, n_rows):
        from kivy.uix.gridlayout import GridLayout
        for ori in GridLayout.orientation.options:
            assert [(0, 0)] == self.compute_layout(n_children=1, ori=ori, n_cols=n_cols, n_rows=n_rows)

    @pytest.mark.parametrize('n_cols, n_rows', [(3, None), (None, 1), (3, 1)])
    @pytest.mark.parametrize('ori', 'lr-tb lr-bt tb-lr bt-lr'.split())
    def test_3x1_lr(self, ori, n_cols, n_rows):
        assert [(0, 0), (100, 0), (200, 0)] == self.compute_layout(n_children=3, ori=ori, n_cols=n_cols, n_rows=n_rows)

    @pytest.mark.parametrize('n_cols, n_rows', [(3, None), (None, 1), (3, 1)])
    @pytest.mark.parametrize('ori', 'rl-tb rl-bt tb-rl bt-rl'.split())
    def test_3x1_rl(self, ori, n_cols, n_rows):
        assert [(200, 0), (100, 0), (0, 0)] == self.compute_layout(n_children=3, ori=ori, n_cols=n_cols, n_rows=n_rows)

    @pytest.mark.parametrize('n_cols, n_rows', [(1, None), (None, 3), (1, 3)])
    @pytest.mark.parametrize('ori', 'tb-lr tb-rl lr-tb rl-tb'.split())
    def test_1x3_tb(self, ori, n_cols, n_rows):
        assert [(0, 200), (0, 100), (0, 0)] == self.compute_layout(n_children=3, ori=ori, n_cols=n_cols, n_rows=n_rows)

    @pytest.mark.parametrize('n_cols, n_rows', [(1, None), (None, 3), (1, 3)])
    @pytest.mark.parametrize('ori', 'bt-lr bt-rl lr-bt rl-bt'.split())
    def test_1x3_bt(self, ori, n_cols, n_rows):
        assert [(0, 0), (0, 100), (0, 200)] == self.compute_layout(n_children=3, ori=ori, n_cols=n_cols, n_rows=n_rows)

    @pytest.mark.parametrize('n_cols, n_rows', [(2, None), (None, 2), (2, 2)])
    def test_2x2_lr_tb(self, n_cols, n_rows):
        assert [(0, 100), (100, 100), (0, 0), (100, 0)] == self.compute_layout(n_children=4, ori='lr-tb', n_cols=n_cols, n_rows=n_rows)

    @pytest.mark.parametrize('n_cols, n_rows', [(2, None), (None, 2), (2, 2)])
    def test_2x2_lr_bt(self, n_cols, n_rows):
        assert [(0, 0), (100, 0), (0, 100), (100, 100)] == self.compute_layout(n_children=4, ori='lr-bt', n_cols=n_cols, n_rows=n_rows)

    @pytest.mark.parametrize('n_cols, n_rows', [(2, None), (None, 2), (2, 2)])
    def test_2x2_rl_tb(self, n_cols, n_rows):
        assert [(100, 100), (0, 100), (100, 0), (0, 0)] == self.compute_layout(n_children=4, ori='rl-tb', n_cols=n_cols, n_rows=n_rows)

    @pytest.mark.parametrize('n_cols, n_rows', [(2, None), (None, 2), (2, 2)])
    def test_2x2_rl_bt(self, n_cols, n_rows):
        assert [(100, 0), (0, 0), (100, 100), (0, 100)] == self.compute_layout(n_children=4, ori='rl-bt', n_cols=n_cols, n_rows=n_rows)

    @pytest.mark.parametrize('n_cols, n_rows', [(2, None), (None, 2), (2, 2)])
    def test_2x2_tb_lr(self, n_cols, n_rows):
        assert [(0, 100), (0, 0), (100, 100), (100, 0)] == self.compute_layout(n_children=4, ori='tb-lr', n_cols=n_cols, n_rows=n_rows)

    @pytest.mark.parametrize('n_cols, n_rows', [(2, None), (None, 2), (2, 2)])
    def test_2x2_tb_rl(self, n_cols, n_rows):
        assert [(100, 100), (100, 0), (0, 100), (0, 0)] == self.compute_layout(n_children=4, ori='tb-rl', n_cols=n_cols, n_rows=n_rows)

    @pytest.mark.parametrize('n_cols, n_rows', [(2, None), (None, 2), (2, 2)])
    def test_2x2_bt_lr(self, n_cols, n_rows):
        assert [(0, 0), (0, 100), (100, 0), (100, 100)] == self.compute_layout(n_children=4, ori='bt-lr', n_cols=n_cols, n_rows=n_rows)

    @pytest.mark.parametrize('n_cols, n_rows', [(2, None), (None, 2), (2, 2)])
    def test_2x2_bt_rl(self, n_cols, n_rows):
        assert [(100, 0), (100, 100), (0, 0), (0, 100)] == self.compute_layout(n_children=4, ori='bt-rl', n_cols=n_cols, n_rows=n_rows)