import re, os, stat, io
from xdg.Exceptions import (ParsingError, DuplicateGroupError, NoGroupError,
import xdg.Locale
from xdg.util import u
Validate the contents, raising :class:`~xdg.Exceptions.ValidationError`
        if there is anything amiss.
        
        report can be 'All' / 'Warnings' / 'Errors'
        