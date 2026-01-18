from collections import OrderedDict
from itertools import chain
class AttrODict(OrderedDict):
    """Ordered dictionary with attribute access (e.g. for tab completion)"""

    def __dir__(self):
        return self.keys()

    def __delattr__(self, name):
        del self[name]

    def __getattr__(self, name):
        return self[name] if not name.startswith('_') else super(AttrODict, self).__getattr__(name)

    def __setattr__(self, name, value):
        if name.startswith('_'):
            return super(AttrODict, self).__setattr__(name, value)
        self[name] = value