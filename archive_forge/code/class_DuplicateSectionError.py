from collections.abc import MutableMapping
from collections import ChainMap as _ChainMap
import functools
import io
import itertools
import os
import re
import sys
import warnings
class DuplicateSectionError(Error):
    """Raised when a section is repeated in an input source.

    Possible repetitions that raise this exception are: multiple creation
    using the API or in strict parsers when a section is found more than once
    in a single input file, string or dictionary.
    """

    def __init__(self, section, source=None, lineno=None):
        msg = [repr(section), ' already exists']
        if source is not None:
            message = ['While reading from ', repr(source)]
            if lineno is not None:
                message.append(' [line {0:2d}]'.format(lineno))
            message.append(': section ')
            message.extend(msg)
            msg = message
        else:
            msg.insert(0, 'Section ')
        Error.__init__(self, ''.join(msg))
        self.section = section
        self.source = source
        self.lineno = lineno
        self.args = (section, source, lineno)