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
class Console(RelativeLayout):
    """Console interface

    This widget is created by create_console(), when the module is loaded.
    During that time, you can add addons on the console to extend the
    functionalities, or add your own application stats / debugging module.
    """
    addons = [ConsoleAddonSelect, ConsoleAddonFps, ConsoleAddonWidgetPanel, ConsoleAddonWidgetTree, ConsoleAddonBreadcrumb]
    mode = OptionProperty('docked', options=['docked', 'floated'])
    widget = ObjectProperty(None, allownone=True)
    inspect_enabled = BooleanProperty(False)
    activated = BooleanProperty(False)

    def __init__(self, **kwargs):
        self.win = kwargs.pop('win', None)
        super(Console, self).__init__(**kwargs)
        self.avoid_bring_to_top = False
        with self.canvas.before:
            self.gcolor = Color(1, 0, 0, 0.25)
            PushMatrix()
            self.gtransform = Transform(Matrix())
            self.grect = Rectangle(size=(0, 0))
            PopMatrix()
        Clock.schedule_interval(self.update_widget_graphics, 0)
        self._toolbar = {'left': [], 'panels': [], 'right': []}
        self._addons = []
        self._panel = None
        for addon in self.addons:
            instance = addon(self)
            self._addons.append(instance)
        self._init_toolbar()
        self._panel = self._toolbar['panels'][0]
        self._panel.state = 'down'
        self._panel.cb_activate()

    def _init_toolbar(self):
        toolbar = self.ids.toolbar
        for key in ('left', 'panels', 'right'):
            if key == 'right':
                toolbar.add_widget(Widget())
            for el in self._toolbar[key]:
                toolbar.add_widget(el)
            if key != 'right':
                toolbar.add_widget(ConsoleAddonSeparator())

    @classmethod
    def register_addon(cls, addon):
        cls.addons.append(addon)

    def add_toolbar_widget(self, widget, right=False):
        """Add a widget in the top left toolbar of the Console.
        Use `right=True` if you wanna add the widget at the right instead.
        """
        key = 'right' if right else 'left'
        self._toolbar[key].append(widget)

    def remove_toolbar_widget(self, widget):
        """Remove a widget from the toolbar
        """
        self.ids.toolbar.remove_widget(widget)

    def add_panel(self, name, cb_activate, cb_deactivate, cb_refresh=None):
        """Add a new panel in the Console.

        - `cb_activate` is a callable that will be called when the panel is
          activated by the user.

        - `cb_deactivate` is a callable that will be called when the panel is
          deactivated or when the console will hide.

        - `cb_refresh` is an optional callable that is called if the user
          click again on the button for display the panel

        When activated, it's up to the panel to display a content in the
        Console by using :meth:`set_content`.
        """
        btn = ConsoleToggleButton(text=name)
        btn.cb_activate = cb_activate
        btn.cb_deactivate = cb_deactivate
        btn.cb_refresh = cb_refresh
        btn.bind(on_press=self._activate_panel)
        self._toolbar['panels'].append(btn)

    def _activate_panel(self, instance):
        if self._panel != instance:
            self._panel.cb_deactivate()
            self._panel.state = 'normal'
            self.ids.content.clear_widgets()
            self._panel = instance
            self._panel.cb_activate()
            self._panel.state = 'down'
        else:
            self._panel.state = 'down'
            if self._panel.cb_refresh:
                self._panel.cb_refresh()

    def set_content(self, content):
        """Replace the Console content with a new one.
        """
        self.ids.content.clear_widgets()
        self.ids.content.add_widget(content)

    def on_touch_down(self, touch):
        ret = super(Console, self).on_touch_down(touch)
        if ('button' not in touch.profile or touch.button == 'left') and (not ret) and self.inspect_enabled:
            self.highlight_at(*touch.pos)
            if touch.is_double_tap:
                self.inspect_enabled = False
            ret = True
        else:
            ret = self.collide_point(*touch.pos)
        return ret

    def on_touch_move(self, touch):
        ret = super(Console, self).on_touch_move(touch)
        if not ret and self.inspect_enabled:
            self.highlight_at(*touch.pos)
            ret = True
        return ret

    def on_touch_up(self, touch):
        ret = super(Console, self).on_touch_up(touch)
        if not ret and self.inspect_enabled:
            ret = True
        return ret

    def on_window_children(self, win, children):
        if self.avoid_bring_to_top or not self.activated:
            return
        self.avoid_bring_to_top = True
        win.remove_widget(self)
        win.add_widget(self)
        self.avoid_bring_to_top = False

    def highlight_at(self, x, y):
        """Select a widget from a x/y window coordinate.
        This is mostly used internally when Select mode is activated
        """
        widget = None
        win_children = self.win.children
        children = chain((c for c in reversed(win_children) if isinstance(c, ModalView)), (c for c in reversed(win_children) if not isinstance(c, ModalView)))
        for child in children:
            if child is self:
                continue
            widget = self.pick(child, x, y)
            if widget:
                break
        self.highlight_widget(widget)

    def highlight_widget(self, widget, *largs):
        self.widget = widget
        if not widget:
            self.grect.size = (0, 0)

    def update_widget_graphics(self, *l):
        if not self.activated:
            return
        if self.widget is None:
            self.grect.size = (0, 0)
            return
        self.grect.size = self.widget.size
        matrix = self.widget.get_window_matrix()
        if self.gtransform.matrix.get() != matrix.get():
            self.gtransform.matrix = matrix

    def pick(self, widget, x, y):
        """Pick a widget at x/y, given a root `widget`
        """
        ret = None
        if hasattr(widget, 'visible') and (not widget.visible):
            return ret
        if widget.collide_point(x, y):
            ret = widget
            x2, y2 = widget.to_local(x, y)
            for child in reversed(widget.children):
                ret = self.pick(child, x2, y2) or ret
        return ret

    def on_activated(self, instance, activated):
        if activated:
            self._activate_console()
        else:
            self._deactivate_console()

    def _activate_console(self):
        if self not in self.win.children:
            self.win.add_widget(self)
        self.y = 0
        for addon in self._addons:
            addon.activate()
        Logger.info('Console: console activated')

    def _deactivate_console(self):
        for addon in self._addons:
            addon.deactivate()
        self.grect.size = (0, 0)
        self.y = -self.height
        self.widget = None
        self.inspect_enabled = False
        self._window_node = None
        Logger.info('Console: console deactivated')

    def keyboard_shortcut(self, win, scancode, *largs):
        modifiers = largs[-1]
        if scancode == 101 and modifiers == ['ctrl']:
            self.activated = not self.activated
            if self.activated:
                self.inspect_enabled = True
            return True
        elif scancode == 27:
            if self.inspect_enabled:
                self.inspect_enabled = False
                return True
            if self.activated:
                self.activated = False
                return True
        if not self.activated or not self.widget:
            return
        if scancode == 273:
            self.widget = self.widget.parent
        elif scancode == 274:
            filtered_children = [c for c in self.widget.children if not isinstance(c, Console)]
            if filtered_children:
                self.widget = filtered_children[0]
        elif scancode == 276:
            parent = self.widget.parent
            filtered_children = [c for c in parent.children if not isinstance(c, Console)]
            index = filtered_children.index(self.widget)
            index = max(0, index - 1)
            self.widget = filtered_children[index]
        elif scancode == 275:
            parent = self.widget.parent
            filtered_children = [c for c in parent.children if not isinstance(c, Console)]
            index = filtered_children.index(self.widget)
            index = min(len(filtered_children) - 1, index + 1)
            self.widget = filtered_children[index]