from collections import OrderedDict
import numpy as _np
class OtherOptionSpace(object):
    """The parameter space for general option"""

    def __init__(self, entities):
        self.entities = [OtherOptionEntity(e) for e in entities]

    @classmethod
    def from_tvm(cls, x):
        return cls([e.val for e in x.entities])

    def __len__(self):
        return len(self.entities)

    def __repr__(self):
        return 'OtherOption(%s) len=%d' % (self.entities, len(self))