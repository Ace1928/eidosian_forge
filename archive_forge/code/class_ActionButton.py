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
class ActionButton(Button, ActionItem):
    """
    ActionButton class, see module documentation for more information.

    The text color, width and size_hint_x are set manually via the Kv language
    file. It covers a lot of cases: with/without an icon, with/without a group
    and takes care of the padding between elements.

    You don't have much control over these properties, so if you want to
    customize its appearance, we suggest you create you own button
    representation. You can do this by creating a class that subclasses an
    existing widget and an :class:`ActionItem`::

        class MyOwnActionButton(Button, ActionItem):
            pass

    You can then create your own style using the Kv language.
    """
    icon = StringProperty(None, allownone=True)
    '\n    Source image to use when the Button is part of the ActionBar. If the\n    Button is in a group, the text will be preferred.\n\n    :attr:`icon` is a :class:`~kivy.properties.StringProperty` and defaults\n    to None.\n    '