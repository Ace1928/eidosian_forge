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
class ImageRenderer(MimetypeRenderer):
    """
    Base class for all static image renderers
    """

    def __init__(self, mime_type, b64_encode=False, format=None, width=None, height=None, scale=None, engine='auto'):
        self.mime_type = mime_type
        self.b64_encode = b64_encode
        self.format = format
        self.width = width
        self.height = height
        self.scale = scale
        self.engine = engine

    def to_mimebundle(self, fig_dict):
        image_bytes = to_image(fig_dict, format=self.format, width=self.width, height=self.height, scale=self.scale, validate=False, engine=self.engine)
        if self.b64_encode:
            image_str = base64.b64encode(image_bytes).decode('utf8')
        else:
            image_str = image_bytes.decode('utf8')
        return {self.mime_type: image_str}