import sys
import os
import re
import warnings
import types
import unicodedata
class image(General, Inline, Element):

    def astext(self):
        return self.get('alt', '')