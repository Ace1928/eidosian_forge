import os
from jedi.api import classes
from jedi.api.strings import StringName, get_quote_ending
from jedi.api.helpers import match
from jedi.inference.helpers import get_str_or_none
class PathName(StringName):
    api_type = 'path'