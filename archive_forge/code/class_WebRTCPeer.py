from __future__ import absolute_import
import os
import logging
from urllib.request import urlopen
from traitlets import (
from ipywidgets import DOMWidget, Image, Video, Audio, register, widget_serialization
import ipywebrtc._version
import traitlets
@register
class WebRTCPeer(DOMWidget):
    """A peer-to-peer webrtc connection"""
    _model_module = Unicode('jupyter-webrtc').tag(sync=True)
    _view_module = Unicode('jupyter-webrtc').tag(sync=True)
    _view_name = Unicode('WebRTCPeerView').tag(sync=True)
    _model_name = Unicode('WebRTCPeerModel').tag(sync=True)
    _view_module_version = Unicode(semver_range_frontend).tag(sync=True)
    _model_module_version = Unicode(semver_range_frontend).tag(sync=True)
    stream_local = Instance(MediaStream, allow_none=True).tag(sync=True, **widget_serialization)
    stream_remote = Instance(MediaStream, allow_none=True).tag(sync=True, **widget_serialization)
    id_local = Unicode('').tag(sync=True)
    id_remote = Unicode('').tag(sync=True)
    connected = Bool(False, read_only=True).tag(sync=True)
    failed = Bool(False, read_only=True).tag(sync=True)

    def connect(self):
        self.send({'msg': 'connect'})