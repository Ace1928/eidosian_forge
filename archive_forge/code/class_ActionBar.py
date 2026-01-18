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
class ActionBar(BoxLayout):
    """
    ActionBar class, which acts as the main container for an
    :class:`ActionView` instance. The ActionBar determines the overall
    styling aspects of the bar. :class:`ActionItem`\\s are not added to
    this class directly, but to the contained :class:`ActionView` instance.

    :Events:
        `on_previous`
            Fired when action_previous of action_view is pressed.

    Please see the module documentation for more information.
    """
    action_view = ObjectProperty(None)
    '\n    action_view of the ActionBar.\n\n    :attr:`action_view` is an :class:`~kivy.properties.ObjectProperty` and\n    defaults to None or the last ActionView instance added to the ActionBar.\n    '
    background_color = ColorProperty([1, 1, 1, 1])
    '\n    Background color, in the format (r, g, b, a).\n\n    :attr:`background_color` is a :class:`~kivy.properties.ColorProperty` and\n    defaults to [1, 1, 1, 1].\n\n    .. versionchanged:: 2.0.0\n        Changed from :class:`~kivy.properties.ListProperty` to\n        :class:`~kivy.properties.ColorProperty`.\n    '
    background_image = StringProperty('atlas://data/images/defaulttheme/action_bar')
    "\n    Background image of the ActionBars default graphical representation.\n\n    :attr:`background_image` is a :class:`~kivy.properties.StringProperty`\n    and defaults to 'atlas://data/images/defaulttheme/action_bar'.\n    "
    border = ListProperty([2, 2, 2, 2])
    '\n    The border to be applied to the :attr:`background_image`.\n\n    :attr:`border` is a :class:`~kivy.properties.ListProperty` and defaults to\n    [2, 2, 2, 2]\n    '
    __events__ = ('on_previous',)

    def __init__(self, **kwargs):
        super(ActionBar, self).__init__(**kwargs)
        self._stack_cont_action_view = []
        self._emit_previous = partial(self.dispatch, 'on_previous')

    def add_widget(self, widget, *args, **kwargs):
        """
        .. versionchanged:: 2.1.0
            Renamed argument `view` to `widget`.
        """
        if isinstance(widget, ContextualActionView):
            self._stack_cont_action_view.append(widget)
            if widget.action_previous is not None:
                widget.action_previous.unbind(on_release=self._emit_previous)
                widget.action_previous.bind(on_release=self._emit_previous)
            self.clear_widgets()
            super(ActionBar, self).add_widget(widget, *args, **kwargs)
        elif isinstance(widget, ActionView):
            self.action_view = widget
            super(ActionBar, self).add_widget(widget, *args, **kwargs)
        else:
            raise ActionBarException('ActionBar can only add ContextualActionView or ActionView')

    def on_previous(self, *args):
        self._pop_contextual_action_view()

    def _pop_contextual_action_view(self):
        """Remove the current ContextualActionView and display either the
           previous one or the ActionView.
        """
        self._stack_cont_action_view.pop()
        self.clear_widgets()
        if self._stack_cont_action_view == []:
            super(ActionBar, self).add_widget(self.action_view)
        else:
            super(ActionBar, self).add_widget(self._stack_cont_action_view[-1])