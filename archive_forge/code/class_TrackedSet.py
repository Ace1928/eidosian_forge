from functools import wraps
from keras.src.backend.common.global_state import get_global_attribute
from keras.src.backend.common.global_state import set_global_attribute
from keras.src.utils import python_utils
class TrackedSet(set):

    def __init__(self, values=None, tracker=None):
        self.tracker = tracker
        if tracker and values:
            values = {tracker.track(v) for v in values}
        super().__init__(values or [])

    def add(self, value):
        if self.tracker:
            self.tracker.track(value)
        super().add(value)

    def update(self, values):
        if self.tracker:
            values = [self.tracker.track(v) for v in values]
        super().update(values)

    def remove(self, value):
        if self.tracker:
            self.tracker.untrack(value)
        super().remove(value)

    def pop(self):
        value = super().pop()
        if self.tracker:
            self.tracker.untrack(value)
        return value

    def clear(self):
        if self.tracker:
            for value in self:
                self.tracker.untrack(value)
        super().clear()