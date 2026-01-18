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
class LegacyReader:

    def __init__(self, plugin_instance: PluginV3, **kwargs):
        self.instance = plugin_instance
        self.last_index = 0
        self.closed = False
        if type(self.instance).__name__ == 'PillowPlugin' and kwargs.get('pilmode') is not None:
            kwargs['mode'] = kwargs['pilmode']
            del kwargs['pilmode']
        self.read_args = kwargs

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

    def get_length(self):
        return self.instance.properties(index=...).n_images

    def get_data(self, index):
        self.last_index = index
        img = self.instance.read(index=index, **self.read_args)
        metadata = self.instance.metadata(index=index, exclude_applied=False)
        return Array(img, metadata)

    def get_next_data(self):
        return self.get_data(self.last_index + 1)

    def set_image_index(self, index):
        self.last_index = index - 1

    def get_meta_data(self, index=None):
        return self.instance.metadata(index=index, exclude_applied=False)

    def iter_data(self):
        for idx, img in enumerate(self.instance.iter()):
            metadata = self.instance.metadata(index=idx, exclude_applied=False)
            yield Array(img, metadata)

    def __iter__(self):
        return self.iter_data()

    def __len__(self):
        return self.get_length()