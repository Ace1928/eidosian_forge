from base64 import b64encode
import io
import json
import pathlib
import uuid
from ipykernel.comm import Comm
from IPython.display import display, Javascript, HTML
from matplotlib import is_interactive
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import _Backend, CloseEvent, NavigationToolbar2
from .backend_webagg_core import (
from .backend_webagg_core import (  # noqa: F401 # pylint: disable=W0611
class NavigationIPy(NavigationToolbar2WebAgg):
    toolitems = [(text, tooltip_text, _FONT_AWESOME_CLASSES[image_file], name_of_method) for text, tooltip_text, image_file, name_of_method in NavigationToolbar2.toolitems + (('Download', 'Download plot', 'download', 'download'),) if image_file in _FONT_AWESOME_CLASSES]