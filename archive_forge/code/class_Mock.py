from unittest import TestCase
import patiencediff
from .. import multiparent, tests
class Mock:

    def __init__(self, **kwargs):
        self.__dict__ = kwargs