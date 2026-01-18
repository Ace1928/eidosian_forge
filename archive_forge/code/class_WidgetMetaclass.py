from kivy.event import EventDispatcher
from kivy.eventmanager import (
from kivy.factory import Factory
from kivy.properties import (
from kivy.graphics import (
from kivy.graphics.transformation import Matrix
from kivy.base import EventLoop
from kivy.lang import Builder
from kivy.context import get_current_context
from kivy.weakproxy import WeakProxy
from functools import partial
from itertools import islice
class WidgetMetaclass(type):
    """Metaclass to automatically register new widgets for the
    :class:`~kivy.factory.Factory`.

    .. warning::
        This metaclass is used by the Widget. Do not use it directly!
    """

    def __init__(mcs, name, bases, attrs):
        super(WidgetMetaclass, mcs).__init__(name, bases, attrs)
        Factory.register(name, cls=mcs)