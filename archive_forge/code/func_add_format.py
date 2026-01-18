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
def add_format(self, iio_format, overwrite=False):
    """add_format(format, overwrite=False)

        Register a format, so that imageio can use it. If a format with the
        same name already exists, an error is raised, unless overwrite is True,
        in which case the current format is replaced.
        """
    warnings.warn('`FormatManager` is deprecated and it will be removed in ImageIO v3.To migrate `FormatManager.add_format` add the plugin directly to `iio.config.known_plugins`.', DeprecationWarning, stacklevel=2)
    if not isinstance(iio_format, Format):
        raise ValueError('add_format needs argument to be a Format object')
    elif not overwrite and iio_format.name in self.get_format_names():
        raise ValueError(f'A Format named {iio_format.name} is already registered, use `overwrite=True` to replace.')
    config = PluginConfig(name=iio_format.name.upper(), class_name=iio_format.__class__.__name__, module_name=iio_format.__class__.__module__, is_legacy=True, install_name='unknown', legacy_args={'name': iio_format.name, 'description': iio_format.description, 'extensions': ' '.join(iio_format.extensions), 'modes': iio_format.modes})
    known_plugins[config.name] = config
    for extension in iio_format.extensions:
        ext = FileExtension(extension=extension, priority=[config.name], name='Unique Format', description=f'A format inserted at runtime. It is being read by the `{config.name}` plugin.')
        known_extensions.setdefault(extension, list()).append(ext)