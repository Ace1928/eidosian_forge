import re
import os
from ast import literal_eval
from functools import partial
from copy import copy
from kivy import kivy_data_dir
from kivy.config import Config
from kivy.utils import platform
from kivy.graphics.texture import Texture
from kivy.core import core_select_lib
from kivy.core.text.text_layout import layout_text, LayoutWord
from kivy.resources import resource_find, resource_add_path
from kivy.compat import PY2
from kivy.setupconfig import USE_SDL2, USE_PANGOFT2
from kivy.logger import Logger
@staticmethod
def add_font(font_context, filename, autocreate=True, family=None):
    """Add a font file to a named font context. If `autocreate` is true,
        the context will be created if it does not exist (this is the
        default). You can specify the `family` argument (string) to skip
        auto-detecting the font family name.

        .. warning::

            The `family` argument is slated for removal if the underlying
            implementation can be fixed, It is offered as a way to optimize
            startup time for deployed applications (it avoids opening the
            file with FreeType2 to determine its family name). To use this,
            first load the font file without specifying `family`, and
            hardcode the returned (autodetected) `family` value in your font
            context initialization.

        .. versionadded:: 1.11.0

        .. note:: This feature requires the Pango text provider.
        """
    raise NotImplementedError('No font_context support in text provider')