import re
from html.parser import HTMLParser
from html import entities
import pyglet
from pyglet.text.formats import structured
def _parse_color(value):
    if value.startswith('#'):
        return _hex_color(int(value[1:], 16))
    else:
        try:
            return _color_names[value.lower()]
        except KeyError:
            raise ValueError()