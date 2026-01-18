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
class SphinxGalleryHtmlRenderer(HtmlRenderer):

    def __init__(self, connected=True, config=None, auto_play=False, post_script=None, animation_opts=None):
        super(SphinxGalleryHtmlRenderer, self).__init__(connected=connected, full_html=False, requirejs=False, global_init=False, config=config, auto_play=auto_play, post_script=post_script, animation_opts=animation_opts)

    def to_mimebundle(self, fig_dict):
        from plotly.io import to_html
        if self.requirejs:
            include_plotlyjs = 'require'
            include_mathjax = False
        elif self.connected:
            include_plotlyjs = 'cdn'
            include_mathjax = 'cdn'
        else:
            include_plotlyjs = True
            include_mathjax = 'cdn'
        html = to_html(fig_dict, config=self.config, auto_play=self.auto_play, include_plotlyjs=include_plotlyjs, include_mathjax=include_mathjax, full_html=self.full_html, animation_opts=self.animation_opts, default_width='100%', default_height=525, validate=False)
        return {'text/html': html}