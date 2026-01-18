from contextlib import ExitStack
import copy
import itertools
from numbers import Integral, Number
from cycler import cycler
import numpy as np
import matplotlib as mpl
from . import (_api, _docstring, backend_tools, cbook, collections, colors,
from .lines import Line2D
from .patches import Circle, Rectangle, Ellipse, Polygon
from .transforms import TransformedPatchPath, Affine2D
class SubplotTool(Widget):
    """
    A tool to adjust the subplot params of a `.Figure`.
    """

    def __init__(self, targetfig, toolfig):
        """
        Parameters
        ----------
        targetfig : `~matplotlib.figure.Figure`
            The figure instance to adjust.
        toolfig : `~matplotlib.figure.Figure`
            The figure instance to embed the subplot tool into.
        """
        self.figure = toolfig
        self.targetfig = targetfig
        toolfig.subplots_adjust(left=0.2, right=0.9)
        toolfig.suptitle('Click on slider to adjust subplot param')
        self._sliders = []
        names = ['left', 'bottom', 'right', 'top', 'wspace', 'hspace']
        for name, ax in zip(names, toolfig.subplots(len(names) + 1)):
            ax.set_navigate(False)
            slider = Slider(ax, name, 0, 1, valinit=getattr(targetfig.subplotpars, name))
            slider.on_changed(self._on_slider_changed)
            self._sliders.append(slider)
        toolfig.axes[-1].remove()
        self.sliderleft, self.sliderbottom, self.sliderright, self.slidertop, self.sliderwspace, self.sliderhspace = self._sliders
        for slider in [self.sliderleft, self.sliderbottom, self.sliderwspace, self.sliderhspace]:
            slider.closedmax = False
        for slider in [self.sliderright, self.slidertop]:
            slider.closedmin = False
        self.sliderleft.slidermax = self.sliderright
        self.sliderright.slidermin = self.sliderleft
        self.sliderbottom.slidermax = self.slidertop
        self.slidertop.slidermin = self.sliderbottom
        bax = toolfig.add_axes([0.8, 0.05, 0.15, 0.075])
        self.buttonreset = Button(bax, 'Reset')
        self.buttonreset.on_clicked(self._on_reset)

    def _on_slider_changed(self, _):
        self.targetfig.subplots_adjust(**{slider.label.get_text(): slider.val for slider in self._sliders})
        if self.drawon:
            self.targetfig.canvas.draw()

    def _on_reset(self, event):
        with ExitStack() as stack:
            stack.enter_context(cbook._setattr_cm(self, drawon=False))
            for slider in self._sliders:
                stack.enter_context(cbook._setattr_cm(slider, drawon=False, eventson=False))
            for slider in self._sliders:
                slider.reset()
        if self.drawon:
            event.canvas.draw()
        self._on_slider_changed(None)