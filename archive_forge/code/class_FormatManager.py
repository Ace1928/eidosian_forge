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
class FormatManager(object):
    """
    The FormatManager is a singleton plugin factory.

    The format manager supports getting a format object using indexing (by
    format name or extension). When used as an iterator, this object
    yields all registered format objects.

    See also :func:`.help`.
    """

    @property
    def _formats(self):
        available_formats = list()
        for config in known_plugins.values():
            with contextlib.suppress(ImportError):
                if config.is_legacy and config.format is not None:
                    available_formats.append(config)
        return available_formats

    def __repr__(self):
        return f'<imageio.FormatManager with {len(self._formats)} registered formats>'

    def __iter__(self):
        return iter((x.format for x in self._formats))

    def __len__(self):
        return len(self._formats)

    def __str__(self):
        ss = []
        for config in self._formats:
            ext = config.legacy_args['extensions']
            desc = config.legacy_args['description']
            s = f'{config.name} - {desc} [{ext}]'
            ss.append(s)
        return '\n'.join(ss)

    def __getitem__(self, name):
        warnings.warn('The usage of `FormatManager` is deprecated and it will be removed in Imageio v3. Use `iio.imopen` instead.', DeprecationWarning, stacklevel=2)
        if not isinstance(name, str):
            raise ValueError('Looking up a format should be done by name or by extension.')
        if name == '':
            raise ValueError('No format matches the empty string.')
        if Path(name).is_file():
            try:
                return imopen(name, 'r', legacy_mode=True)._format
            except ValueError:
                pass
        config = _get_config(name.upper())
        try:
            return config.format
        except ImportError:
            raise ImportError(f'The `{config.name}` format is not installed. Use `pip install imageio[{config.install_name}]` to install it.')

    def sort(self, *names):
        """sort(name1, name2, name3, ...)

        Sort the formats based on zero or more given names; a format with
        a name that matches one of the given names will take precedence
        over other formats. A match means an equal name, or ending with
        that name (though the former counts higher). Case insensitive.

        Format preference will match the order of the given names: using
        ``sort('TIFF', '-FI', '-PIL')`` would prefer the FreeImage formats
        over the Pillow formats, but prefer TIFF even more. Each time
        this is called, the starting point is the default format order,
        and calling ``sort()`` with no arguments will reset the order.

        Be aware that using the function can affect the behavior of
        other code that makes use of imageio.

        Also see the ``IMAGEIO_FORMAT_ORDER`` environment variable.
        """
        warnings.warn('`FormatManager` is deprecated and it will be removed in ImageIO v3. Migrating `FormatManager.sort` depends on your use-case:\n\t- modify `iio.config.known_plugins` to specify the search order for unrecognized formats.\n\t- modify `iio.config.known_extensions[<extension>].priority` to control a specific extension.', DeprecationWarning, stacklevel=2)
        for name in names:
            if not isinstance(name, str):
                raise TypeError('formats.sort() accepts only string names.')
            if any((c in name for c in '.,')):
                raise ValueError('Names given to formats.sort() should not contain dots `.` or commas `,`.')
        should_reset = len(names) == 0
        if should_reset:
            names = _original_order
        sane_names = [name.strip().upper() for name in names if name != '']
        flat_extensions = [ext for ext_list in known_extensions.values() for ext in ext_list]
        for extension in flat_extensions:
            if should_reset:
                extension.reset()
                continue
            for name in reversed(sane_names):
                for plugin in [x for x in extension.default_priority]:
                    if plugin.endswith(name):
                        extension.priority.remove(plugin)
                        extension.priority.insert(0, plugin)
        old_order = known_plugins.copy()
        known_plugins.clear()
        for name in sane_names:
            plugin = old_order.pop(name, None)
            if plugin is not None:
                known_plugins[name] = plugin
        known_plugins.update(old_order)

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

    def search_read_format(self, request):
        """search_read_format(request)

        Search a format that can read a file according to the given request.
        Returns None if no appropriate format was found. (used internally)
        """
        try:
            return imopen(request, request.mode.io_mode, legacy_mode=True)._format
        except AttributeError:
            warnings.warn('ImageIO now uses a v3 plugin when reading this format. Please migrate to the v3 API (preferred) or use imageio.v2.', DeprecationWarning, stacklevel=2)
            return None
        except ValueError:
            return None

    def search_write_format(self, request):
        """search_write_format(request)

        Search a format that can write a file according to the given request.
        Returns None if no appropriate format was found. (used internally)
        """
        try:
            return imopen(request, request.mode.io_mode, legacy_mode=True)._format
        except AttributeError:
            warnings.warn('ImageIO now uses a v3 plugin when writing this format. Please migrate to the v3 API (preferred) or use imageio.v2.', DeprecationWarning, stacklevel=2)
            return None
        except ValueError:
            return None

    def get_format_names(self):
        """Get the names of all registered formats."""
        warnings.warn('`FormatManager` is deprecated and it will be removed in ImageIO v3.To migrate `FormatManager.get_format_names` use `iio.config.known_plugins.keys()` instead.', DeprecationWarning, stacklevel=2)
        return [f.name for f in self._formats]

    def show(self):
        """Show a nicely formatted list of available formats"""
        print(self)