import re
import warnings
from numbers import Number
from pathlib import Path
from typing import Dict
import numpy as np
from imageio.core.legacy_plugin_wrapper import LegacyPlugin
from imageio.core.util import Array
from imageio.core.v3_plugin_api import PluginV3
from . import formats
from .config import known_extensions, known_plugins
from .core import RETURN_BYTES
from .core.imopen import imopen
class LegacyWriter:

    def __init__(self, plugin_instance: PluginV3, **kwargs):
        self.instance = plugin_instance
        self.last_index = 0
        self.closed = False
        if type(self.instance).__name__ == 'PillowPlugin' and 'pilmode' in kwargs:
            kwargs['mode'] = kwargs['pilmode']
            del kwargs['pilmode']
        self.write_args = kwargs

    def close(self):
        if not self.closed:
            self.instance.close()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __del__(self):
        self.close()

    @property
    def request(self):
        return self.instance.request

    @property
    def format(self):
        raise TypeError("V3 Plugins don't have a format.")

    def append_data(self, im, meta=None):
        if meta is not None:
            warnings.warn("V3 Plugins currently don't have a uniform way to write metadata, so any metadata is ignored.")
        return self.instance.write(im, **self.write_args)

    def set_meta_data(self, meta):
        raise NotImplementedError("V3 Plugins don't have a uniform way to write metadata (yet).")