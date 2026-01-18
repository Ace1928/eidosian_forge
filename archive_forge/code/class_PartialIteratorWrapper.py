import re
import sys
import warnings
class PartialIteratorWrapper:

    def __init__(self, wsgi_iterator):
        self.iterator = wsgi_iterator

    def __iter__(self):
        return IteratorWrapper(self.iterator, None)