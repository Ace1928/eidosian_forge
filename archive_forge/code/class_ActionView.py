from kivy.uix.boxlayout import BoxLayout
from kivy.uix.dropdown import DropDown
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.checkbox import CheckBox
from kivy.uix.spinner import Spinner
from kivy.uix.label import Label
from kivy.config import Config
from kivy.properties import ObjectProperty, NumericProperty, BooleanProperty, \
from kivy.metrics import sp
from kivy.lang import Builder
from functools import partial
class ActionView(BoxLayout):
    """
    ActionView class, see module documentation for more information.
    """
    action_previous = ObjectProperty(None)
    '\n    Previous button for an ActionView.\n\n    :attr:`action_previous` is an :class:`~kivy.properties.ObjectProperty`\n    and defaults to None.\n    '
    background_color = ColorProperty([1, 1, 1, 1])
    '\n    Background color in the format (r, g, b, a).\n\n    :attr:`background_color` is a :class:`~kivy.properties.ColorProperty` and\n    defaults to [1, 1, 1, 1].\n\n    .. versionchanged:: 2.0.0\n        Changed from :class:`~kivy.properties.ListProperty` to\n        :class:`~kivy.properties.ColorProperty`.\n    '
    background_image = StringProperty('atlas://data/images/defaulttheme/action_view')
    "\n    Background image of an ActionViews default graphical representation.\n\n    :attr:`background_image` is a :class:`~kivy.properties.StringProperty`\n    and defaults to 'atlas://data/images/defaulttheme/action_view'.\n    "
    use_separator = BooleanProperty(False)
    '\n    Specify whether to use a separator before every ActionGroup or not.\n\n    :attr:`use_separator` is a :class:`~kivy.properties.BooleanProperty` and\n    defaults to False.\n    '
    overflow_group = ObjectProperty(None)
    '\n    Widget to be used for the overflow.\n\n    :attr:`overflow_group` is an :class:`~kivy.properties.ObjectProperty` and\n    defaults to an instance of :class:`ActionOverflow`.\n    '

    def __init__(self, **kwargs):
        self._list_action_items = []
        self._list_action_group = []
        super(ActionView, self).__init__(**kwargs)
        self._state = ''
        if not self.overflow_group:
            self.overflow_group = ActionOverflow(use_separator=self.use_separator)

    def on_action_previous(self, instance, value):
        self._list_action_items.insert(0, value)

    def add_widget(self, widget, index=0, *args, **kwargs):
        """
        .. versionchanged:: 2.1.0
            Renamed argument `action_item` to `widget`.
        """
        if widget is None:
            return
        if not isinstance(widget, ActionItem):
            raise ActionBarException('ActionView only accepts ActionItem (got {!r}'.format(widget))
        elif isinstance(widget, ActionOverflow):
            self.overflow_group = widget
            widget.use_separator = self.use_separator
        elif isinstance(widget, ActionGroup):
            self._list_action_group.append(widget)
            widget.use_separator = self.use_separator
        elif isinstance(widget, ActionPrevious):
            self.action_previous = widget
        else:
            super(ActionView, self).add_widget(widget, index, *args, **kwargs)
            if index == 0:
                index = len(self._list_action_items)
            self._list_action_items.insert(index, widget)

    def on_use_separator(self, instance, value):
        for group in self._list_action_group:
            group.use_separator = value
        if self.overflow_group:
            self.overflow_group.use_separator = value

    def remove_widget(self, widget, *args, **kwargs):
        super(ActionView, self).remove_widget(widget, *args, **kwargs)
        if isinstance(widget, ActionOverflow):
            for item in widget.list_action_item:
                if item in self._list_action_items:
                    self._list_action_items.remove(item)
        if widget in self._list_action_items:
            self._list_action_items.remove(widget)

    def _clear_all(self):
        lst = self._list_action_items[:]
        self.clear_widgets()
        for group in self._list_action_group:
            group.clear_widgets()
        self.overflow_group.clear_widgets()
        self.overflow_group.list_action_item = []
        self._list_action_items = lst

    def _layout_all(self):
        super_add = super(ActionView, self).add_widget
        self._state = 'all'
        self._clear_all()
        if not self.action_previous.parent:
            super_add(self.action_previous)
        if len(self._list_action_items) > 1:
            for child in self._list_action_items[1:]:
                child.inside_group = False
                super_add(child)
        for group in self._list_action_group:
            if group.mode == 'spinner':
                super_add(group)
                group.show_group()
            else:
                if group.list_action_item != []:
                    super_add(ActionSeparator())
                for child in group.list_action_item:
                    child.inside_group = False
                    super_add(child)
        self.overflow_group.show_default_items(self)

    def _layout_group(self):
        super_add = super(ActionView, self).add_widget
        self._state = 'group'
        self._clear_all()
        if not self.action_previous.parent:
            super_add(self.action_previous)
        if len(self._list_action_items) > 1:
            for child in self._list_action_items[1:]:
                super_add(child)
                child.inside_group = False
        for group in self._list_action_group:
            super_add(group)
            group.show_group()
        self.overflow_group.show_default_items(self)

    def _layout_random(self):
        super_add = super(ActionView, self).add_widget
        self._state = 'random'
        self._clear_all()
        hidden_items = []
        hidden_groups = []
        total_width = 0
        if not self.action_previous.parent:
            super_add(self.action_previous)
        width = self.width - self.overflow_group.pack_width - self.action_previous.minimum_width
        if len(self._list_action_items):
            for child in self._list_action_items[1:]:
                if child.important:
                    if child.pack_width + total_width < width:
                        super_add(child)
                        child.inside_group = False
                        total_width += child.pack_width
                    else:
                        hidden_items.append(child)
                else:
                    hidden_items.append(child)
        if total_width < self.width:
            for group in self._list_action_group:
                if group.pack_width + total_width + group.separator_width < width:
                    super_add(group)
                    group.show_group()
                    total_width += group.pack_width + group.separator_width
                else:
                    hidden_groups.append(group)
        group_index = len(self.children) - 1
        if total_width < self.width:
            for child in hidden_items[:]:
                if child.pack_width + total_width < width:
                    super_add(child, group_index)
                    total_width += child.pack_width
                    child.inside_group = False
                    hidden_items.remove(child)
        extend_hidden = hidden_items.extend
        for group in hidden_groups:
            extend_hidden(group.list_action_item)
        overflow_group = self.overflow_group
        if hidden_items != []:
            over_add = super(overflow_group.__class__, overflow_group).add_widget
            for child in hidden_items:
                over_add(child)
            overflow_group.show_group()
            if not self.overflow_group.parent:
                super_add(overflow_group)

    def on_width(self, width, *args):
        total_width = 0
        for child in self._list_action_items:
            total_width += child.pack_width
        for group in self._list_action_group:
            for child in group.list_action_item:
                total_width += child.pack_width
        if total_width <= self.width:
            if self._state != 'all':
                self._layout_all()
            return
        total_width = 0
        for child in self._list_action_items:
            total_width += child.pack_width
        for group in self._list_action_group:
            total_width += group.pack_width
        if total_width < self.width:
            if self._state != 'group':
                self._layout_group()
            return
        self._layout_random()