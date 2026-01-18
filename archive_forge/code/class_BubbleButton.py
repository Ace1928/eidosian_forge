from kivy.uix.image import Image
from kivy.uix.scatter import Scatter
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.button import Button
from kivy.properties import ObjectProperty
from kivy.properties import StringProperty
from kivy.properties import OptionProperty
from kivy.properties import ListProperty
from kivy.properties import BooleanProperty
from kivy.properties import ColorProperty
from kivy.properties import NumericProperty
from kivy.properties import ReferenceListProperty
from kivy.base import EventLoop
from kivy.metrics import dp
class BubbleButton(Button):
    """A button intended for use in a BubbleContent widget.
    You can use a "normal" button class, but it will not look good unless the
    background is changed.

    Rather use this BubbleButton widget that is already defined and provides a
    suitable background for you.
    """
    pass