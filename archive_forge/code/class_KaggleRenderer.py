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
class KaggleRenderer(HtmlRenderer):
    """
    Renderer to display interactive figures in Kaggle Notebooks.

    Same as NotebookRenderer but with connected=True so that the plotly.js
    bundle is loaded from a CDN rather than being embedded in the notebook.

    This renderer is enabled by default when running in a Kaggle notebook.

    mime type: 'text/html'
    """

    def __init__(self, config=None, auto_play=False, post_script=None, animation_opts=None):
        super(KaggleRenderer, self).__init__(connected=True, full_html=False, requirejs=True, global_init=True, config=config, auto_play=auto_play, post_script=post_script, animation_opts=animation_opts)