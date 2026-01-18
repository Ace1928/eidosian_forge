from pathlib import Path
import numpy as np
from ..config import known_extensions
from .request import InitializationError, IOMode
from .v3_plugin_api import ImageProperties, PluginV3
def _legacy_default_index(format):
    if format._name == 'FFMPEG':
        index = Ellipsis
    elif format._name == 'GIF-PIL':
        index = Ellipsis
    else:
        index = 0
    return index