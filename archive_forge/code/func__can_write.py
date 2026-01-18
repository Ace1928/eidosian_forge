import sys
import warnings
import contextlib
import numpy as np
from pathlib import Path
from . import Array, asarray
from .request import ImageMode
from ..config import known_plugins, known_extensions, PluginConfig, FileExtension
from ..config.plugins import _original_order
from .imopen import imopen
def _can_write(self, request):
    """Check if Plugin can write to ImageResource.

        Parameters
        ----------
        request : Request
            A request that can be used to access the ImageResource and obtain
            metadata about it.

        Returns
        -------
        can_read : bool
            True if the plugin can write to the ImageResource, False otherwise.

        """
    return None