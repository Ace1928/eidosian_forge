import sys
import os
import re
import warnings
import types
import unicodedata
class BackLinkable:

    def add_backref(self, refid):
        self['backrefs'].append(refid)