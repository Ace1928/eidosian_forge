from typing import ClassVar, List, Sequence
from twisted.python.util import FancyEqMixin

        Emit a VT102 control sequence that will set up all the attributes this
        formatting state has set.

        @return: A string containing VT102 control sequences that mimic this
            formatting state.
        