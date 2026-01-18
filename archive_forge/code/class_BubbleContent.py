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
class BubbleContent(BoxLayout):
    """A styled BoxLayout that can be used as the content widget of a Bubble.

    .. versionchanged:: 2.2.0
    The graphical appearance of :class:`BubbleContent` is now based on it's
    own properties :attr:`background_image`, :attr:`background_color`,
    :attr:`border` and :attr:`border_auto_scale`. The parent widget properties
    are no longer considered. This makes the BubbleContent a standalone themed
    BoxLayout.
    """
    background_color = ColorProperty([1, 1, 1, 1])
    'Background color, in the format (r, g, b, a). To use it you have to set\n    :attr:`background_image` first.\n\n    .. versionadded:: 2.2.0\n\n    :attr:`background_color` is a :class:`~kivy.properties.ColorProperty` and\n    defaults to [1, 1, 1, 1].\n    '
    background_image = StringProperty('atlas://data/images/defaulttheme/bubble')
    "Background image of the bubble.\n\n    .. versionadded:: 2.2.0\n\n    :attr:`background_image` is a :class:`~kivy.properties.StringProperty` and\n    defaults to 'atlas://data/images/defaulttheme/bubble'.\n    "
    border = ListProperty([16, 16, 16, 16])
    'Border used for :class:`~kivy.graphics.vertex_instructions.BorderImage`\n    graphics instruction. Used with the :attr:`background_image`.\n    It should be used when using custom backgrounds.\n\n    It must be a list of 4 values: (bottom, right, top, left). Read the\n    BorderImage instructions for more information about how to use it.\n\n    .. versionadded:: 2.2.0\n\n    :attr:`border` is a :class:`~kivy.properties.ListProperty` and defaults to\n    (16, 16, 16, 16)\n    '
    border_auto_scale = OptionProperty('both_lower', options=['off', 'both', 'x_only', 'y_only', 'y_full_x_lower', 'x_full_y_lower', 'both_lower'])
    "Specifies the :attr:`kivy.graphics.BorderImage.auto_scale`\n    value on the background BorderImage.\n\n    .. versionadded:: 2.2.0\n\n    :attr:`border_auto_scale` is a\n    :class:`~kivy.properties.OptionProperty` and defaults to\n    'both_lower'.\n    "