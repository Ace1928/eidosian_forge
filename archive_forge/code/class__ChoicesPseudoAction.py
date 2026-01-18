import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
class _ChoicesPseudoAction(Action):

    def __init__(self, name, aliases, help):
        metavar = dest = name
        if aliases:
            metavar += ' (%s)' % ', '.join(aliases)
        sup = super(_SubParsersAction._ChoicesPseudoAction, self)
        sup.__init__(option_strings=[], dest=dest, help=help, metavar=metavar)