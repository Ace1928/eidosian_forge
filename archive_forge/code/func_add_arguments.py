import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def add_arguments(self, actions):
    for action in actions:
        self.add_argument(action)