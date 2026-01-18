import os, time, webbrowser
from .gui import *
from . import smooth
from .vertex import Vertex
from .arrow import Arrow
from .crossings import Crossing, ECrossing
from .colors import Palette
from .dialog import InfoDialog
from .manager import LinkManager
from .viewer import LinkViewer
from .version import version
from .ipython_tools import IPythonTkRoot
def _add_zoom_and_pan(self, style_menu):
    zoom_menu = Tk_.Menu(style_menu, tearoff=0)
    pan_menu = Tk_.Menu(style_menu, tearoff=0)
    if sys.platform == 'darwin':
        zoom_menu.add_command(label='Zoom in    \t+', command=self.zoom_in)
        zoom_menu.add_command(label='Zoom out   \t-', command=self.zoom_out)
        zoom_menu.add_command(label='Zoom to fit\t0', command=self.zoom_to_fit)
        pan_menu.add_command(label='Left  \t' + scut['Left'], command=lambda: self._shift(-5, 0))
        pan_menu.add_command(label='Up    \t' + scut['Up'], command=lambda: self._shift(0, -5))
        pan_menu.add_command(label='Right \t' + scut['Right'], command=lambda: self._shift(5, 0))
        pan_menu.add_command(label='Down  \t' + scut['Down'], command=lambda: self._shift(0, 5))
    else:
        zoom_menu.add_command(label='Zoom in', accelerator='+', command=self.zoom_in)
        zoom_menu.add_command(label='Zoom out', accelerator='-', command=self.zoom_out)
        zoom_menu.add_command(label='Zoom to fit', accelerator='0', command=self.zoom_to_fit)
        pan_menu.add_command(label='Left', accelerator=scut['Left'], command=lambda: self._shift(-5, 0))
        pan_menu.add_command(label='Up', accelerator=scut['Up'], command=lambda: self._shift(0, -5))
        pan_menu.add_command(label='Right', accelerator=scut['Right'], command=lambda: self._shift(5, 0))
        pan_menu.add_command(label='Down', accelerator=scut['Down'], command=lambda: self._shift(0, 5))
    style_menu.add_separator()
    style_menu.add_cascade(label='Zoom', menu=zoom_menu)
    style_menu.add_cascade(label='Pan', menu=pan_menu)