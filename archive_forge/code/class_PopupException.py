from kivy.core.text import DEFAULT_FONT
from kivy.uix.modalview import ModalView
from kivy.properties import (StringProperty, ObjectProperty, OptionProperty,
class PopupException(Exception):
    """Popup exception, fired when multiple content widgets are added to the
    popup.

    .. versionadded:: 1.4.0
    """