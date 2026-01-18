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
class NotebookRenderer(HtmlRenderer):
    """
    Renderer to display interactive figures in the classic Jupyter Notebook.
    This renderer is also useful for notebooks that will be converted to
    HTML using nbconvert/nbviewer as it will produce standalone HTML files
    that include interactive figures.

    This renderer automatically performs global notebook initialization when
    activated.

    mime type: 'text/html'
    """

    def __init__(self, connected=False, config=None, auto_play=False, post_script=None, animation_opts=None):
        super(NotebookRenderer, self).__init__(connected=connected, full_html=False, requirejs=True, global_init=True, config=config, auto_play=auto_play, post_script=post_script, animation_opts=animation_opts)