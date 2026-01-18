import os
from unittest import SkipTest
import param
from IPython.core.completer import IPCompleter
from IPython.display import HTML, publish_display_data
from param import ipython as param_ext
import holoviews as hv
from ..core.dimension import LabelledData
from ..core.options import Store
from ..core.tree import AttrTree
from ..element.comparison import ComparisonTestCase
from ..plotting.renderer import Renderer
from ..util import extension
from .display_hooks import display, png_display, pprint_display, svg_display
from .magics import load_magics
class IPTestCase(ComparisonTestCase):
    """
    This class extends ComparisonTestCase to handle IPython specific
    objects and support the execution of cells and magic.
    """

    def setUp(self):
        super().setUp()
        try:
            import IPython
            from IPython.display import HTML, SVG
            self.ip = IPython.InteractiveShell()
            if self.ip is None:
                raise TypeError()
        except Exception as e:
            raise SkipTest('IPython could not be started') from e
        self.ip.displayhook.flush = lambda: None
        self.addTypeEqualityFunc(HTML, self.skip_comparison)
        self.addTypeEqualityFunc(SVG, self.skip_comparison)

    def skip_comparison(self, obj1, obj2, msg):
        pass

    def get_object(self, name):
        obj = self.ip._object_find(name).obj
        if obj is None:
            raise self.failureException(f'Could not find object {name}')
        return obj

    def cell(self, line):
        """Run an IPython cell"""
        self.ip.run_cell(line, silent=True)

    def cell_magic(self, *args, **kwargs):
        """Run an IPython cell magic"""
        self.ip.run_cell_magic(*args, **kwargs)

    def line_magic(self, *args, **kwargs):
        """Run an IPython line magic"""
        self.ip.run_line_magic(*args, **kwargs)