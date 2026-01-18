import numpy as np
import pandas as pd
import pytest
from unittest import TestCase, SkipTest
from hvplot.util import (
class TestDynamicArgs(TestCase):

    def setUp(self):
        try:
            import panel as pn
        except:
            raise SkipTest('panel not available')

    def test_dynamic_and_static(self):
        import panel as pn
        from ..util import process_dynamic_args
        x = 'sepal_width'
        y = pn.widgets.Select(name='y', value='sepal_length', options=['sepal_length', 'petal_length'])
        kind = pn.widgets.Select(name='kind', value='scatter', options=['bivariate', 'scatter'])
        dynamic, arg_deps, arg_names = process_dynamic_args(x, y, kind)
        assert 'x' not in dynamic
        assert 'y' in dynamic
        assert arg_deps == []

    def test_dynamic_kwds(self):
        import panel as pn
        from ..util import process_dynamic_args
        x = 'sepal_length'
        y = 'sepal_width'
        kind = 'scatter'
        color = pn.widgets.ColorPicker(value='#ff0000')
        dynamic, arg_deps, arg_names = process_dynamic_args(x, y, kind, c=color)
        assert 'x' not in dynamic
        assert 'c' in dynamic
        assert arg_deps == []

    def test_fn_kwds(self):
        import panel as pn
        from ..util import process_dynamic_args
        x = 'sepal_length'
        y = 'sepal_width'
        kind = 'scatter'
        by_species = pn.widgets.Checkbox(name='By species')
        color = pn.widgets.ColorPicker(value='#ff0000')

        @pn.depends(by_species.param.value, color.param.value)
        def by_species_fn(by_species, color):
            return 'species' if by_species else color
        dynamic, arg_deps, arg_names = process_dynamic_args(x, y, kind, c=by_species_fn)
        assert dynamic == {}
        assert arg_names == ['c', 'c']
        assert len(arg_deps) == 2