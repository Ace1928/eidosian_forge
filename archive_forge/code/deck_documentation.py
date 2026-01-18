import os
import sys
from .json_tools import JSONMixin
from .layer import Layer
from ..io.html import deck_to_html
from ..settings import settings as pydeck_settings
from .view import View
from .view_state import ViewState
from .base_map_provider import BaseMapProvider
from .map_styles import DARK, get_from_map_identifier
Write a file and loads it to an iframe, if in a Jupyter environment;
        otherwise, write a file and optionally open it in a web browser

        Parameters
        ----------
        filename : str, default None
            Name of the file.
        open_browser : bool, default False
            Whether a browser window will open or not after write.
        notebook_display : bool, default None
            Display the HTML output in an iframe if True. Set to True automatically if rendering in Jupyter.
        iframe_width : str or int, default '100%'
            Width of Jupyter notebook iframe in pixels, if rendered in a Jupyter environment.
        iframe_height : int, default 500
            Height of Jupyter notebook iframe in pixels, if rendered in Jupyter or Colab.
        as_string : bool, default False
            Returns HTML as a string, if True and ``filename`` is None.
        css_background_color : str, default None
            Background color for visualization, specified as a string in any format accepted for CSS colors.

        Returns
        -------
        str
            Returns absolute path of the file
        