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
class ConsoleAddonWidgetTreeImpl(TreeView):
    selected_widget = ObjectProperty(None, allownone=True)
    __events__ = ('on_select_widget',)

    def __init__(self, **kwargs):
        super(ConsoleAddonWidgetTreeImpl, self).__init__(**kwargs)
        self.update_scroll = Clock.create_trigger(self._update_scroll)

    def find_node_by_widget(self, widget):
        for node in self.iterate_all_nodes():
            if not node.parent_node:
                continue
            try:
                if node.widget == widget:
                    return node
            except ReferenceError:
                pass
        return None

    def update_selected_widget(self, widget):
        if widget:
            node = self.find_node_by_widget(widget)
            if node:
                self.select_node(node, False)
                while node and isinstance(node, TreeViewWidget):
                    if not node.is_open:
                        self.toggle_node(node)
                    node = node.parent_node

    def on_selected_widget(self, inst, widget):
        if widget:
            self.update_selected_widget(widget)
            self.update_scroll()

    def select_node(self, node, select_widget=True):
        super(ConsoleAddonWidgetTreeImpl, self).select_node(node)
        if select_widget:
            try:
                self.dispatch('on_select_widget', node.widget.__self__)
            except ReferenceError:
                pass

    def on_select_widget(self, widget):
        pass

    def _update_scroll(self, *args):
        node = self._selected_node
        if not node:
            return
        self.parent.scroll_to(node)