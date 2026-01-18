from contextlib import contextmanager
from kombu.utils.objects import cached_property
@cached_property
def State(self):
    return self.app.subclass_with_self(self.state_cls, reverse='events.State')