import re
from html.parser import HTMLParser
from html import entities
import pyglet
from pyglet.text.formats import structured
def _hex_color(val):
    return [val >> 16 & 255, val >> 8 & 255, val & 255, 255]