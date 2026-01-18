from kivy.properties import AliasProperty, StringProperty, ColorProperty
from kivy.uix.behaviors import ToggleButtonBehavior
from kivy.uix.widget import Widget
Color is used for tinting the default graphical representation
    of checkbox and radio button (images).

    Color is in the format (r, g, b, a).

    .. versionadded:: 1.10.0

    :attr:`color` is a
    :class:`~kivy.properties.ColorProperty` and defaults to
    '[1, 1, 1, 1]'.

    .. versionchanged:: 2.0.0
        Changed from :class:`~kivy.properties.ListProperty` to
        :class:`~kivy.properties.ColorProperty`.
    