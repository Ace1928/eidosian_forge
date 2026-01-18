import html
import itertools
from contextlib import closing
from inspect import isclass
from io import StringIO
from pathlib import Path
from string import Template
from .. import __version__, config_context
from .fixes import parse_version
def _get_css_style():
    return Path(__file__).with_suffix('.css').read_text(encoding='utf-8')