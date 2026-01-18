from __future__ import unicode_literals
from six import with_metaclass
from collections import defaultdict
import weakref
class _FilterType(with_metaclass(_FilterTypeMeta)):

    def __new__(cls):
        raise NotImplementedError('This class should not be initiated.')