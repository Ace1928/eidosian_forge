import kivy
import weakref
from functools import partial
from itertools import chain
from kivy.logger import Logger
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image
from kivy.uix.treeview import TreeViewNode, TreeView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.modalview import ModalView
from kivy.graphics import Color, Rectangle, PushMatrix, PopMatrix
from kivy.graphics.context_instructions import Transform
from kivy.graphics.transformation import Matrix
from kivy.properties import (ObjectProperty, BooleanProperty, ListProperty,
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.lang import Builder
def _update_widget_tree_node(self, node, widget, is_open=False):
    tree = self.ids.widgettree
    update_nodes = []
    nodes = {}
    for cnode in node.nodes[:]:
        try:
            nodes[cnode.widget] = cnode
        except ReferenceError:
            pass
        tree.remove_node(cnode)
    for child in widget.children:
        if isinstance(child, Console):
            continue
        if child in nodes:
            cnode = tree.add_node(nodes[child], node)
        else:
            cnode = tree.add_node(TreeViewWidget(text=child.__class__.__name__, widget=child.proxy_ref, is_open=is_open), node)
        update_nodes.append((cnode, child))
    return update_nodes