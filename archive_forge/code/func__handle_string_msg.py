from .widget_description import DescriptionStyle, DescriptionWidget
from .valuewidget import ValueWidget
from .widget import CallbackDispatcher, register, widget_serialization
from .widget_core import CoreWidget
from .trait_types import Color, InstanceDict, TypedTuple
from .utils import deprecation
from traitlets import Unicode, Bool, Int
def _handle_string_msg(self, _, content, buffers):
    """Handle a msg from the front-end.

        Parameters
        ----------
        content: dict
            Content of the msg.
        """
    if content.get('event', '') == 'submit':
        self._submission_callbacks(self)