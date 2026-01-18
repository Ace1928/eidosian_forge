from contextlib import contextmanager
from kombu.utils.objects import cached_property
@cached_property
def Dispatcher(self):
    return self.app.subclass_with_self(self.dispatcher_cls, reverse='events.Dispatcher')