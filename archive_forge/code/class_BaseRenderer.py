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
class BaseRenderer(object):
    """
    Base class for all renderers
    """

    def activate(self):
        pass

    def __repr__(self):
        try:
            init_sig = inspect.signature(self.__init__)
            init_args = list(init_sig.parameters.keys())
        except AttributeError:
            argspec = inspect.getargspec(self.__init__)
            init_args = [a for a in argspec.args if a != 'self']
        return '{cls}({attrs})\n{doc}'.format(cls=self.__class__.__name__, attrs=', '.join(('{}={!r}'.format(k, self.__dict__[k]) for k in init_args)), doc=self.__doc__)

    def __hash__(self):
        return hash(repr(self))