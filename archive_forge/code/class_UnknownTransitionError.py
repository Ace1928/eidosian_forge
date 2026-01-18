import sys
import re
import types
import unicodedata
from docutils import utils
from docutils.utils.error_reporting import ErrorOutput
class UnknownTransitionError(StateMachineError):
    pass