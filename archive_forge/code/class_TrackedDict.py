from functools import wraps
from keras.src.backend.common.global_state import get_global_attribute
from keras.src.backend.common.global_state import set_global_attribute
from keras.src.utils import python_utils
class TrackedDict(dict):

    def __init__(self, values=None, tracker=None):
        self.tracker = tracker
        if tracker and values:
            values = {k: tracker.track(v) for k, v in values.items()}
        super().__init__(values or [])

    def __setitem__(self, key, value):
        if self.tracker:
            self.tracker.track(value)
        super().__setitem__(key, value)

    def update(self, mapping):
        if self.tracker:
            mapping = {k: self.tracker.track(v) for k, v in mapping.items()}
        super().update(mapping)

    def pop(self, key, default=None):
        if self.tracker:
            value = super().pop(key, default)
            if value is not default:
                self.tracker.untrack(value)
            return value
        else:
            return super().pop(key, default)

    def popitem(self):
        key, value = super().popitem()
        if self.tracker:
            self.tracker.untrack(value)
        return (key, value)

    def clear(self):
        if self.tracker:
            for value in self.values():
                self.tracker.untrack(value)
        super().clear()