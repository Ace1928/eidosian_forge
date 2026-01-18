from __future__ import absolute_import
import os
import logging
from urllib.request import urlopen
from traitlets import (
from ipywidgets import DOMWidget, Image, Video, Audio, register, widget_serialization
import ipywebrtc._version
import traitlets
@staticmethod
def _facing(facing_mode, audio=True, **kwargs):
    kwargs = dict(kwargs)
    constraints = kwargs.pop('constraints', {})
    if 'audio' not in constraints:
        constraints['audio'] = audio
    if 'video' not in constraints:
        constraints['video'] = {}
    constraints['video']['facingMode'] = facing_mode
    return CameraStream(constraints=constraints, **kwargs)