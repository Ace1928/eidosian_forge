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
class ActionPrevious(BoxLayout, ActionItem):
    """
    ActionPrevious class, see module documentation for more information.
    """
    with_previous = BooleanProperty(True)
    '\n    Specifies whether the previous_icon will be shown or not. Note that it is\n    up to the user to implement the desired behavior using the *on_press* or\n    similar events.\n\n    :attr:`with_previous` is a :class:`~kivy.properties.BooleanProperty` and\n    defaults to True.\n    '
    app_icon = StringProperty(window_icon)
    "\n    Application icon for the ActionView.\n\n    :attr:`app_icon` is a :class:`~kivy.properties.StringProperty`\n    and defaults to the window icon if set, otherwise\n    'data/logo/kivy-icon-32.png'.\n    "
    app_icon_width = NumericProperty(0)
    '\n    Width of app_icon image.\n\n    :attr:`app_icon_width` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 0.\n    '
    app_icon_height = NumericProperty(0)
    '\n    Height of app_icon image.\n\n    :attr:`app_icon_height` is a :class:`~kivy.properties.NumericProperty`\n    and defaults to 0.\n    '
    color = ColorProperty([1, 1, 1, 1])
    '\n    Text color, in the format (r, g, b, a)\n\n    :attr:`color` is a :class:`~kivy.properties.ColorProperty` and defaults\n    to [1, 1, 1, 1].\n\n    .. versionchanged:: 2.0.0\n        Changed from :class:`~kivy.properties.ListProperty` to\n        :class:`~kivy.properties.ColorProperty`.\n    '
    previous_image = StringProperty('atlas://data/images/defaulttheme/previous_normal')
    "\n    Image for the 'previous' ActionButtons default graphical representation.\n\n    :attr:`previous_image` is a :class:`~kivy.properties.StringProperty` and\n    defaults to 'atlas://data/images/defaulttheme/previous_normal'.\n    "
    previous_image_width = NumericProperty(0)
    '\n    Width of previous_image image.\n\n    :attr:`width` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 0.\n    '
    previous_image_height = NumericProperty(0)
    '\n    Height of previous_image image.\n\n    :attr:`app_icon_width` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 0.\n    '
    title = StringProperty('')
    "\n    Title for ActionView.\n\n    :attr:`title` is a :class:`~kivy.properties.StringProperty` and\n    defaults to ''.\n    "
    markup = BooleanProperty(False)
    '\n    If True, the text will be rendered using the\n    :class:`~kivy.core.text.markup.MarkupLabel`: you can change the style of\n    the text using tags. Check the :doc:`api-kivy.core.text.markup`\n    documentation for more information.\n\n    :attr:`markup` is a :class:`~kivy.properties.BooleanProperty` and\n    defaults to False.\n    '

    def __init__(self, **kwargs):
        self.register_event_type('on_press')
        self.register_event_type('on_release')
        super(ActionPrevious, self).__init__(**kwargs)
        if not self.app_icon:
            self.app_icon = 'data/logo/kivy-icon-32.png'

    def on_press(self):
        pass

    def on_release(self):
        pass