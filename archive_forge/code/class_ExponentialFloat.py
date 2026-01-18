from __future__ import print_function, absolute_import, division, unicode_literals
import sys
from .compat import no_limit_int  # NOQA
from ruamel.yaml.anchor import Anchor
class ExponentialFloat(ScalarFloat):

    def __new__(cls, value, width=None, underscore=None):
        return ScalarFloat.__new__(cls, value, width=width, underscore=underscore)