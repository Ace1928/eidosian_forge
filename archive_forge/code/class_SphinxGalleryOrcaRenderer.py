import base64
import json
import webbrowser
import inspect
import os
from os.path import isdir
from plotly import utils, optional_imports
from plotly.io import to_json, to_image, write_image, write_html
from plotly.io._orca import ensure_server
from plotly.io._utils import plotly_cdn_url
from plotly.offline.offline import _get_jconfig, get_plotlyjs
from plotly.tools import return_figure_from_figure_or_data
class SphinxGalleryOrcaRenderer(ExternalRenderer):

    def render(self, fig_dict):
        stack = inspect.stack()
        try:
            filename = stack[3].filename
        except:
            filename = stack[3][1]
        filename_root, _ = os.path.splitext(filename)
        filename_html = filename_root + '.html'
        filename_png = filename_root + '.png'
        figure = return_figure_from_figure_or_data(fig_dict, True)
        _ = write_html(fig_dict, file=filename_html, include_plotlyjs='cdn')
        try:
            write_image(figure, filename_png)
        except (ValueError, ImportError):
            raise ImportError('orca and psutil are required to use the `sphinx-gallery-orca` renderer. See https://plotly.com/python/static-image-export/ for instructions on how to install orca. Alternatively, you can use the `sphinx-gallery` renderer (note that png thumbnails can only be generated with the `sphinx-gallery-orca` renderer).')