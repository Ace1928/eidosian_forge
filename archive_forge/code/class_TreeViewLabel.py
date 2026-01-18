from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.properties import BooleanProperty, ListProperty, ObjectProperty, \
class TreeViewLabel(Label, TreeViewNode):
    """Combines a :class:`~kivy.uix.label.Label` and a :class:`TreeViewNode` to
    create a :class:`TreeViewLabel` that can be used as a text node in the
    tree.

    See module documentation for more information.
    """