from __future__ import absolute_import
import math, sys
class _EmptyDecoratorAndManager(object):

    def __call__(self, x):
        return x

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass