from contextlib import contextmanager
from copy import copy
class ContextDict(dict):

    def __init__(self, context, *args, **kwargs):
        super().__init__(*args, **kwargs)
        context.dicts.append(self)
        self.context = context

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.context.pop()