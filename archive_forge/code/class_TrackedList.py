from functools import wraps
from keras.src.backend.common.global_state import get_global_attribute
from keras.src.backend.common.global_state import set_global_attribute
from keras.src.utils import python_utils
class TrackedList(list):

    def __init__(self, values=None, tracker=None):
        self.tracker = tracker
        if tracker and values:
            values = [tracker.track(v) for v in values]
        super().__init__(values or [])

    def append(self, value):
        if self.tracker:
            self.tracker.track(value)
        super().append(value)

    def insert(self, value):
        if self.tracker:
            self.tracker.track(value)
        super().insert(value)

    def extend(self, values):
        if self.tracker:
            values = [self.tracker.track(v) for v in values]
        super().extend(values)

    def remove(self, value):
        if self.tracker:
            self.tracker.untrack(value)
        try:
            super().remove(value)
        except ValueError:
            python_utils.remove_by_id(self, value)

    def pop(self, index=-1):
        if self.tracker:
            value = self[index]
            self.tracker.untrack(value)
            return super().pop(index)
        else:
            return super().pop(index)

    def clear(self):
        if self.tracker:
            for value in self:
                self.tracker.untrack(value)
        super().clear()

    def __delitem__(self, index):
        value = self[index]
        super().__delitem__(index)
        if self.tracker:
            self.tracker.untrack(value)