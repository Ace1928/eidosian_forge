from __future__ import absolute_import
import os
import logging
from urllib.request import urlopen
from traitlets import (
from ipywidgets import DOMWidget, Image, Video, Audio, register, widget_serialization
import ipywebrtc._version
import traitlets
@register
class WidgetStream(MediaStream):
    """Represents a widget media source.
    """
    _model_name = Unicode('WidgetStreamModel').tag(sync=True)
    _view_name = Unicode('WidgetStreamView').tag(sync=True)
    widget = Instance(DOMWidget, allow_none=False, help='An instance of ipywidgets.DOMWidget that will be the source of the MediaStream.').tag(sync=True, **widget_serialization)
    max_fps = Int(None, allow_none=True, help='(int, default None) The maximum amount of frames per second to capture, or only on new data when the valeus is None.').tag(sync=True)
    _html2canvas_start_streaming = Bool(False).tag(sync=True)

    @validate('max_fps')
    def _valid_fps(self, proposal):
        if proposal['value'] is not None and proposal['value'] < 0:
            raise TraitError('max_fps attribute must be a positive integer')
        return proposal['value']