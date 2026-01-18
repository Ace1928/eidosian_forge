from kivy.config import Config
from kivy.logger import Logger
from kivy.input import providers
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
def _get_provider_map(self):
    """Iterates through all registered input provider names and finds the
        respective MotionEvent subclass for each. Returns a dict of MotionEvent
        subclasses mapped to their provider name.
        """
    provider_map = {}
    for input_provider in MotionEventFactory.list():
        if not hasattr(providers, input_provider):
            continue
        p = getattr(providers, input_provider)
        for m in p.__all__:
            event = getattr(p, m)
            if issubclass(event, MotionEvent):
                provider_map[event] = input_provider
    return provider_map