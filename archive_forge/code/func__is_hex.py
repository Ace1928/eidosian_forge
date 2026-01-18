import json
import math
import os
from jinja2 import Template
from branca.element import ENV, Figure, JavascriptLink, MacroElement
from branca.utilities import legend_scaler
def _is_hex(x):
    return x.startswith('#') and len(x) == 7