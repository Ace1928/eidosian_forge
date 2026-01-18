import os
from os.path import dirname, join, exists, abspath
from kivy.clock import Clock
from kivy.compat import PY2
from kivy.properties import ObjectProperty, NumericProperty, \
from kivy.lang import Builder
from kivy.utils import get_hex_from_color, get_color_from_hex
from kivy.uix.widget import Widget
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.image import AsyncImage, Image
from kivy.uix.videoplayer import VideoPlayer
from kivy.uix.anchorlayout import AnchorLayout
from kivy.animation import Animation
from kivy.logger import Logger
from docutils.parsers import rst
from docutils.parsers.rst import roles
from docutils import nodes, frontend, utils
from docutils.parsers.rst import Directive, directives
from docutils.parsers.rst.roles import set_classes
def _load_from_text(self, *largs):
    try:
        self.content.clear_widgets()
        self.anchors_widgets = []
        self.refs_assoc = {}
        document = utils.new_document('Document', self._settings)
        text = self.text
        if PY2 and type(text) is str:
            text = text.decode('utf-8')
        self._parser.parse(text, document)
        visitor = _Visitor(self, document)
        document.walkabout(visitor)
        self.title = visitor.title or 'No title'
    except:
        Logger.exception('Rst: error while loading text')