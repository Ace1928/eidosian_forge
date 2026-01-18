from __future__ import annotations
import aifc
import audioop
import base64
import collections
import hashlib
import hmac
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from .audio import AudioData, get_flac_converter
from .exceptions import (
class MicrophoneStream(object):

    def __init__(self, pyaudio_stream):
        self.pyaudio_stream = pyaudio_stream

    def read(self, size):
        return self.pyaudio_stream.read(size, exception_on_overflow=False)

    def close(self):
        try:
            if not self.pyaudio_stream.is_stopped():
                self.pyaudio_stream.stop_stream()
        finally:
            self.pyaudio_stream.close()