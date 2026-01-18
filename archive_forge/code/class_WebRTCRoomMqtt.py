from __future__ import absolute_import
import os
import logging
from urllib.request import urlopen
from traitlets import (
from ipywidgets import DOMWidget, Image, Video, Audio, register, widget_serialization
import ipywebrtc._version
import traitlets
@register
class WebRTCRoomMqtt(WebRTCRoom):
    """Use a mqtt server to connect to other peers"""
    _model_name = Unicode('WebRTCRoomMqttModel').tag(sync=True)
    server = Unicode('wss://iot.eclipse.org:443/ws').tag(sync=True)