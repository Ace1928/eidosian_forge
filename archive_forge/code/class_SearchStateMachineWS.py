import sys
import re
import types
import unicodedata
from docutils import utils
from docutils.utils.error_reporting import ErrorOutput
class SearchStateMachineWS(_SearchOverride, StateMachineWS):
    """`StateMachineWS` which uses `re.search()` instead of `re.match()`."""
    pass