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
class ConsoleAddonBreadcrumbView(RelativeLayout):
    widget = ObjectProperty(None, allownone=True)
    parents = []

    def on_widget(self, instance, value):
        stack = self.ids.stack
        prefs = [btn.widget_ref() for btn in self.parents]
        if value in prefs:
            index = prefs.index(value)
            for btn in self.parents:
                btn.state = 'normal'
            self.parents[index].state = 'down'
            return
        stack.clear_widgets()
        if not value:
            return
        widget = value
        parents = []
        while True:
            btn = ConsoleButton(text=widget.__class__.__name__)
            btn.widget_ref = weakref.ref(widget)
            btn.bind(on_release=self.highlight_widget)
            parents.append(btn)
            if widget == widget.parent:
                break
            widget = widget.parent
        for btn in reversed(parents):
            stack.add_widget(btn)
        self.ids.sv.scroll_x = 1
        self.parents = parents
        btn.state = 'down'

    def highlight_widget(self, instance):
        self.console.widget = instance.widget_ref()