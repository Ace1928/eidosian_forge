import enum
from contextlib import nullcontext
from pyomo.common.deprecation import deprecated
@deprecated("The clone counter has been removed and will always return 0.\n\nBeginning with Pyomo5 expressions, expression cloning (detangling) no\nlonger occurs automatically within expression generation.  As a result,\nthe 'clone counter' has lost its utility and is no longer supported.\nThis context manager will always report 0.", version='6.4.3')
class clone_counter(nullcontext):
    """Context manager for counting cloning events.

    This context manager counts the number of times that the
    :func:`clone_expression <pyomo.core.expr.current.clone_expression>`
    function is executed.
    """
    _count = 0

    def __init__(self):
        super().__init__(enter_result=self)

    @property
    def count(self):
        """A property that returns the clone count value."""
        return clone_counter._count