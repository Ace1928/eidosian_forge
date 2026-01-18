import docutils.parsers
import docutils.statemachine
from docutils.parsers.rst import states
from docutils import frontend, nodes, Component
from docutils.transforms import universal
class DirectiveError(Exception):
    """
    Store a message and a system message level.

    To be thrown from inside directive code.

    Do not instantiate directly -- use `Directive.directive_error()`
    instead!
    """

    def __init__(self, level, message):
        """Set error `message` and `level`"""
        Exception.__init__(self)
        self.level = level
        self.msg = message