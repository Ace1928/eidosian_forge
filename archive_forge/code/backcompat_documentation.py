import codecs
import copy
import json
import os
from urllib.parse import unquote
import bs4
from . import mf2_classes
from .dom_helpers import get_children
from .mf_helpers import unordered_list
add modern classnames for older mf1 classnames

    returns a copy of el and does not modify the original
    