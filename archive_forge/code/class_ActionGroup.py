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
class ActionGroup(ActionItem, Button):
    """
    ActionGroup class, see module documentation for more information.
    """
    use_separator = BooleanProperty(False)
    '\n    Specifies whether to use a separator after/before this group or not.\n\n    :attr:`use_separator` is a :class:`~kivy.properties.BooleanProperty` and\n    defaults to False.\n    '
    separator_image = StringProperty('atlas://data/images/defaulttheme/separator')
    "\n    Background Image for an ActionSeparator in an ActionView.\n\n    :attr:`separator_image` is a :class:`~kivy.properties.StringProperty`\n    and defaults to 'atlas://data/images/defaulttheme/separator'.\n    "
    separator_width = NumericProperty(0)
    '\n    Width of the ActionSeparator in an ActionView.\n\n    :attr:`separator_width` is a :class:`~kivy.properties.NumericProperty`\n    and defaults to 0.\n    '
    mode = OptionProperty('normal', options=('normal', 'spinner'))
    "\n    Sets the current mode of an ActionGroup. If mode is 'normal', the\n    ActionGroups children will be displayed normally if there is enough\n    space, otherwise they will be displayed in a spinner. If mode is\n    'spinner', then the children will always be displayed in a spinner.\n\n    :attr:`mode` is an :class:`~kivy.properties.OptionProperty` and defaults\n    to 'normal'.\n    "
    dropdown_width = NumericProperty(0)
    "\n    If non zero, provides the width for the associated DropDown. This is\n    useful when some items in the ActionGroup's DropDown are wider than usual\n    and you don't want to make the ActionGroup widget itself wider.\n\n    :attr:`dropdown_width` is a :class:`~kivy.properties.NumericProperty`\n    and defaults to 0.\n\n    .. versionadded:: 1.10.0\n    "
    is_open = BooleanProperty(False)
    'By default, the DropDown is not open. Set to True to open it.\n\n    :attr:`is_open` is a :class:`~kivy.properties.BooleanProperty` and\n    defaults to False.\n    '

    def __init__(self, **kwargs):
        self.list_action_item = []
        self._list_overflow_items = []
        super(ActionGroup, self).__init__(**kwargs)
        self._is_open = False
        self._dropdown = ActionDropDown()
        self._dropdown.bind(attach_to=lambda ins, value: setattr(self, '_is_open', True if value else False))
        self.bind(on_release=lambda *args: setattr(self, 'is_open', True))
        self._dropdown.bind(on_dismiss=lambda *args: setattr(self, 'is_open', False))

    def on_is_open(self, instance, value):
        if value and (not self._is_open):
            self._toggle_dropdown()
            self._dropdown.open(self)
            return
        if not value and self._is_open:
            self._dropdown.dismiss()

    def _toggle_dropdown(self, *largs):
        ddn = self._dropdown
        ddn.size_hint_x = None
        if not ddn.container:
            return
        children = ddn.container.children
        if children:
            ddn.width = self.dropdown_width or max(self.width, max((c.pack_width for c in children)))
        else:
            ddn.width = self.width
        for item in children:
            item.size_hint_y = None
            item.height = max([self.height, sp(48)])
            item.bind(on_release=ddn.dismiss)

    def add_widget(self, widget, *args, **kwargs):
        """
        .. versionchanged:: 2.1.0
            Renamed argument `item` to `widget`.
        """
        if isinstance(widget, ActionSeparator):
            super(ActionGroup, self).add_widget(widget, *args, **kwargs)
            return
        if not isinstance(widget, ActionItem):
            raise ActionBarException('ActionGroup only accepts ActionItem')
        self.list_action_item.append(widget)

    def show_group(self):
        self.clear_widgets()
        for item in self._list_overflow_items + self.list_action_item:
            item.inside_group = True
            self._dropdown.add_widget(item)

    def clear_widgets(self, *args, **kwargs):
        self._dropdown.clear_widgets(*args, **kwargs)