from __future__ import absolute_import
import os
import logging
from urllib.request import urlopen
from traitlets import (
from ipywidgets import DOMWidget, Image, Video, Audio, register, widget_serialization
import ipywebrtc._version
import traitlets
@traitlets.default('image')
def _default_image(self):
    return Image(width=self._width, height=self._height, format=self.format)