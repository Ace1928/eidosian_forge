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
class _ToctreeVisitor(nodes.NodeVisitor):

    def __init__(self, *largs):
        self.toctree = self.current = []
        self.queue = []
        self.text = ''
        nodes.NodeVisitor.__init__(self, *largs)

    def push(self, tree):
        self.queue.append(tree)
        self.current = tree

    def pop(self):
        self.current = self.queue.pop()

    def dispatch_visit(self, node):
        cls = node.__class__
        if cls is nodes.section:
            section = {'ids': node['ids'], 'names': node['names'], 'title': '', 'children': []}
            if isinstance(self.current, dict):
                self.current['children'].append(section)
            else:
                self.current.append(section)
            self.push(section)
        elif cls is nodes.title:
            self.text = ''
        elif cls is nodes.Text:
            self.text += node

    def dispatch_departure(self, node):
        cls = node.__class__
        if cls is nodes.section:
            self.pop()
        elif cls is nodes.title:
            self.current['title'] = self.text