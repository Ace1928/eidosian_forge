from binascii import a2b_base64
from io import BytesIO
import pytest
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
import numpy as np
from IPython.core.getipython import get_ipython
from IPython.core.interactiveshell import InteractiveShell
from IPython.core.display import _PNG, _JPEG
from .. import pylabtools as pt
from IPython.testing import decorators as dec
class TestPylabSwitch(object):

    class Shell(InteractiveShell):

        def init_history(self):
            """Sets up the command history, and starts regular autosaves."""
            self.config.HistoryManager.hist_file = ':memory:'
            super().init_history()

        def enable_gui(self, gui):
            pass

    def setup(self):
        import matplotlib

        def act_mpl(backend):
            matplotlib.rcParams['backend'] = backend
        self._saved_rcParams = matplotlib.rcParams
        self._saved_rcParamsOrig = matplotlib.rcParamsOrig
        matplotlib.rcParams = dict(backend='QtAgg')
        matplotlib.rcParamsOrig = dict(backend='QtAgg')
        self._save_am = pt.activate_matplotlib
        pt.activate_matplotlib = act_mpl
        self._save_ip = pt.import_pylab
        pt.import_pylab = lambda *a, **kw: None
        self._save_cis = backend_inline.configure_inline_support
        backend_inline.configure_inline_support = lambda *a, **kw: None

    def teardown(self):
        pt.activate_matplotlib = self._save_am
        pt.import_pylab = self._save_ip
        backend_inline.configure_inline_support = self._save_cis
        import matplotlib
        matplotlib.rcParams = self._saved_rcParams
        matplotlib.rcParamsOrig = self._saved_rcParamsOrig

    def test_qt(self):
        s = self.Shell()
        gui, backend = s.enable_matplotlib(None)
        assert gui == 'qt'
        assert s.pylab_gui_select == 'qt'
        gui, backend = s.enable_matplotlib('inline')
        assert gui == 'inline'
        assert s.pylab_gui_select == 'qt'
        gui, backend = s.enable_matplotlib('qt')
        assert gui == 'qt'
        assert s.pylab_gui_select == 'qt'
        gui, backend = s.enable_matplotlib('inline')
        assert gui == 'inline'
        assert s.pylab_gui_select == 'qt'
        gui, backend = s.enable_matplotlib()
        assert gui == 'qt'
        assert s.pylab_gui_select == 'qt'

    def test_inline(self):
        s = self.Shell()
        gui, backend = s.enable_matplotlib('inline')
        assert gui == 'inline'
        assert s.pylab_gui_select == None
        gui, backend = s.enable_matplotlib('inline')
        assert gui == 'inline'
        assert s.pylab_gui_select == None
        gui, backend = s.enable_matplotlib('qt')
        assert gui == 'qt'
        assert s.pylab_gui_select == 'qt'

    def test_inline_twice(self):
        """Using '%matplotlib inline' twice should not reset formatters"""
        ip = self.Shell()
        gui, backend = ip.enable_matplotlib('inline')
        assert gui == 'inline'
        fmts = {'png'}
        active_mimes = {_fmt_mime_map[fmt] for fmt in fmts}
        pt.select_figure_formats(ip, fmts)
        gui, backend = ip.enable_matplotlib('inline')
        assert gui == 'inline'
        for mime, f in ip.display_formatter.formatters.items():
            if mime in active_mimes:
                assert Figure in f
            else:
                assert Figure not in f

    def test_qt_gtk(self):
        s = self.Shell()
        gui, backend = s.enable_matplotlib('qt')
        assert gui == 'qt'
        assert s.pylab_gui_select == 'qt'
        gui, backend = s.enable_matplotlib('gtk')
        assert gui == 'qt'
        assert s.pylab_gui_select == 'qt'