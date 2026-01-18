from unittest import TestCase
from unittest import mock
import pytest
import traitlets
import ipywidgets as widgets
from ipywidgets.widgets.widget_templates import LayoutProperties
class TestGridspecLayout(TestCase):
    """test GridspecLayout"""

    def test_init(self):
        with pytest.raises(traitlets.TraitError):
            box = widgets.GridspecLayout()
        with pytest.raises(traitlets.TraitError):
            box = widgets.GridspecLayout(n_rows=-1, n_columns=1)
        box = widgets.GridspecLayout(n_rows=5, n_columns=3)
        assert box.n_rows == 5
        assert box.n_columns == 3
        assert len(box._grid_template_areas) == 5
        assert len(box._grid_template_areas[0]) == 3
        box = widgets.GridspecLayout(1, 2)
        assert box.n_rows == 1
        assert box.n_columns == 2
        with pytest.raises(traitlets.TraitError):
            box = widgets.GridspecLayout(0, 0)

    def test_setitem_index(self):
        box = widgets.GridspecLayout(2, 3)
        button1 = widgets.Button()
        button2 = widgets.Button()
        button3 = widgets.Button()
        button4 = widgets.Button()
        box[0, 0] = button1
        button1_label = button1.layout.grid_area
        assert button1 in box.children
        assert box.layout.grid_template_areas == '"{} . ."\n". . ."'.format(button1_label)
        box[-1, -1] = button2
        button2_label = button2.layout.grid_area
        assert button1_label != button2_label
        assert button2 in box.children
        assert box.layout.grid_template_areas == '"{} . ."\n". . {}"'.format(button1_label, button2_label)
        box[1, 0] = button3
        button3_label = button3.layout.grid_area
        assert button1_label != button3_label
        assert button2_label != button3_label
        assert button3 in box.children
        assert box.layout.grid_template_areas == '"{b1} . ."\n"{b3} . {b2}"'.format(b1=button1_label, b2=button2_label, b3=button3_label)
        box[1, 0] = button4
        button4_label = button4.layout.grid_area
        assert button1_label != button4_label
        assert button2_label != button4_label
        assert button4 in box.children
        assert button3 not in box.children
        assert box.layout.grid_template_areas == '"{b1} . ."\n"{b4} . {b2}"'.format(b1=button1_label, b2=button2_label, b4=button4_label)

    def test_setitem_slices(self):
        box = widgets.GridspecLayout(2, 3)
        button1 = widgets.Button()
        box[:2, 0] = button1
        assert len(box.children) == 1
        assert button1 in box.children
        button1_label = button1.layout.grid_area
        assert box.layout.grid_template_areas == '"{b1} . ."\n"{b1} . ."'.format(b1=button1_label)
        box = widgets.GridspecLayout(2, 3)
        button1 = widgets.Button()
        button2 = widgets.Button()
        box[:2, 1:] = button1
        assert len(box.children) == 1
        assert button1 in box.children
        button1_label = button1.layout.grid_area
        assert box.layout.grid_template_areas == '". {b1} {b1}"\n". {b1} {b1}"'.format(b1=button1_label)
        box[:2, 1:] = button2
        assert len(box.children) == 1
        assert button2 in box.children
        button2_label = button2.layout.grid_area
        assert box.layout.grid_template_areas == '". {b1} {b1}"\n". {b1} {b1}"'.format(b1=button2_label)

    def test_getitem_index(self):
        """test retrieving widget"""
        box = widgets.GridspecLayout(2, 3)
        button1 = widgets.Button()
        box[0, 0] = button1
        assert box[0, 0] is button1

    def test_getitem_slices(self):
        """test retrieving widgets with slices"""
        box = widgets.GridspecLayout(2, 3)
        button1 = widgets.Button()
        box[:2, 0] = button1
        assert box[:2, 0] is button1
        box = widgets.GridspecLayout(2, 3)
        button1 = widgets.Button()
        button2 = widgets.Button()
        box[0, 0] = button1
        box[1, 0] = button2
        assert box[0, 0] is button1
        assert box[1, 0] is button2
        with pytest.raises(TypeError, match='The slice spans'):
            button = box[:2, 0]