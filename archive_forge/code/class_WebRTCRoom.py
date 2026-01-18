from __future__ import absolute_import
import os
import logging
from urllib.request import urlopen
from traitlets import (
from ipywidgets import DOMWidget, Image, Video, Audio, register, widget_serialization
import ipywebrtc._version
import traitlets
@register
class WebRTCRoom(DOMWidget):
    """A 'chatroom', which consists of a list of :`WebRTCPeer` connections
    """
    _model_module = Unicode('jupyter-webrtc').tag(sync=True)
    _view_module = Unicode('jupyter-webrtc').tag(sync=True)
    _model_name = Unicode('WebRTCRoomModel').tag(sync=True)
    _view_module_version = Unicode(semver_range_frontend).tag(sync=True)
    _model_module_version = Unicode(semver_range_frontend).tag(sync=True)
    room = Unicode('room').tag(sync=True)
    stream = Instance(MediaStream, allow_none=True).tag(sync=True, **widget_serialization)
    room_id = Unicode(read_only=True).tag(sync=True)
    nickname = Unicode('anonymous').tag(sync=True)
    peers = List(Instance(WebRTCPeer), [], allow_none=False).tag(sync=True, **widget_serialization)
    streams = List(Instance(MediaStream), [], allow_none=False).tag(sync=True, **widget_serialization)