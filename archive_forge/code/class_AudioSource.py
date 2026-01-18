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
class AudioSource(object):

    def __init__(self):
        raise NotImplementedError('this is an abstract class')

    def __enter__(self):
        raise NotImplementedError('this is an abstract class')

    def __exit__(self, exc_type, exc_value, traceback):
        raise NotImplementedError('this is an abstract class')