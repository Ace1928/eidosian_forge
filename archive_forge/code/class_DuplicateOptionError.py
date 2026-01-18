from collections.abc import MutableMapping
from collections import ChainMap as _ChainMap
import functools
import io
import itertools
import os
import re
import sys
import warnings
class DuplicateOptionError(Error):
    """Raised by strict parsers when an option is repeated in an input source.

    Current implementation raises this exception only when an option is found
    more than once in a single file, string or dictionary.
    """

    def __init__(self, section, option, source=None, lineno=None):
        msg = [repr(option), ' in section ', repr(section), ' already exists']
        if source is not None:
            message = ['While reading from ', repr(source)]
            if lineno is not None:
                message.append(' [line {0:2d}]'.format(lineno))
            message.append(': option ')
            message.extend(msg)
            msg = message
        else:
            msg.insert(0, 'Option ')
        Error.__init__(self, ''.join(msg))
        self.section = section
        self.option = option
        self.source = source
        self.lineno = lineno
        self.args = (section, option, source, lineno)