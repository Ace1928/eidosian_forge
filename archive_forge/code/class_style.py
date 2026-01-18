from __future__ import absolute_import
from __future__ import division
import pythreejs
import os
import time
import warnings
import tempfile
import uuid
import base64
from io import BytesIO as StringIO
import six
import numpy as np
import PIL.Image
import matplotlib.style
import ipywidgets
import IPython
from IPython.display import display
import ipyvolume as ipv
import ipyvolume.embed
from ipyvolume import utils
from . import ui
class style:
    """Static class that mimics a matplotlib module.

    Example:

    >>> import ipyvolume as ipv
    >>> ipv.style.use('light'])
    >>> ipv.style.use('seaborn-darkgrid'])
    >>> ipv.style.use(['seaborn-darkgrid', {'axes.x.color':'orange'}])

    Possible style values:
     * figure.facecolor: background color
     * axes.color: color of the box around the volume/viewport
     * xaxis.color: color of xaxis
     * yaxis.color: color of xaxis
     * zaxis.color: color of xaxis

    """

    @staticmethod
    def use(style):
        """Set the style of the current figure/visualization.

        :param style: matplotlib style name, or dict with values, or a sequence of these, where the last value overrides previous
        :return:
        """

        def valid(value):
            return isinstance(value, six.string_types)

        def translate(mplstyle):
            style = {}
            mapping = [['figure.facecolor', 'background-color'], ['xtick.color', 'axes.x.color'], ['xtick.color', 'axes.z.color'], ['ytick.color', 'axes.y.color'], ['axes.labelcolor', 'axes.label.color'], ['text.color', 'color'], ['axes.edgecolor', 'axes.color']]
            for from_name, to_name in mapping:
                if from_name in mplstyle:
                    value = mplstyle[from_name]
                    if 'color' in from_name:
                        try:
                            value = float(value) * 255
                            value = 'rgb(%d, %d, %d)' % (value, value, value)
                        except:
                            pass
                    utils.nested_setitem(style, to_name, value)
            return style
        if isinstance(style, six.string_types + (dict,)):
            styles = [style]
        else:
            styles = style
        fig = gcf()
        totalstyle = utils.dict_deep_update({}, fig.style)
        for style in styles:
            if isinstance(style, six.string_types):
                if hasattr(ipyvolume.styles, style):
                    style = getattr(ipyvolume.styles, style)
                else:
                    cleaned_style = {key: value for key, value in dict(matplotlib.style.library[style]).items() if valid(value)}
                    style = translate(cleaned_style)
            else:
                pass
            totalstyle = utils.dict_deep_update(totalstyle, style)
        fig = gcf()
        fig.style = totalstyle

    @staticmethod
    def _axes(which=None, **values):
        if which:
            style.use({'axes': {name: values for name in which}})
        else:
            style.use({'axes': values})

    @staticmethod
    def axes_off(which=None):
        """Do not draw the axes, optionally give axis names, e.g. 'xy'."""
        style._axes(which, visible=False)

    @staticmethod
    def axes_on(which=None):
        """Draw the axes, optionally give axis names, e.g. 'xy'."""
        style._axes(which, visible=True)

    @staticmethod
    def box_off():
        """Do not draw the box around the visible volume."""
        style.use({'box': {'visible': False}})

    @staticmethod
    def box_on():
        """Draw a box around the visible volume."""
        style.use({'box': {'visible': True}})

    @staticmethod
    def background_color(color):
        """Set the background color."""
        style.use({'background-color': color})