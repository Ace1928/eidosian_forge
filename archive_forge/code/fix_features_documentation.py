from .feature_base import Feature, Features
from lib2to3 import fixer_base

Warn about features that are not present in Python 2.5, giving a message that
points to the earliest version of Python 2.x (or 3.x, if none) that supports it
