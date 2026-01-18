import warnings
from ._basic import Is
from ._const import Always
from ._datastructures import MatchesListwise, MatchesStructure
from ._higherorder import (
from ._impl import Mismatch
def WarningMessage(category_type, message=None, filename=None, lineno=None, line=None):
    """
    Create a matcher that will match `warnings.WarningMessage`\\s.

    For example, to match captured `DeprecationWarning`s with a message about
    some ``foo`` being replaced with ``bar``:

    .. code-block:: python

       WarningMessage(DeprecationWarning,
                      message=MatchesAll(
                          Contains('foo is deprecated'),
                          Contains('use bar instead')))

    :param type category_type: A warning type, for example `DeprecationWarning`.
    :param message_matcher: A matcher object that will be evaluated against
        warning's message.
    :param filename_matcher: A matcher object that will be evaluated against
        the warning's filename.
    :param lineno_matcher: A matcher object that will be evaluated against the
        warning's line number.
    :param line_matcher: A matcher object that will be evaluated against the
        warning's line of source code.
    """
    category_matcher = Is(category_type)
    message_matcher = message or Always()
    filename_matcher = filename or Always()
    lineno_matcher = lineno or Always()
    line_matcher = line or Always()
    return MatchesStructure(category=Annotate("Warning's category type does not match", category_matcher), message=Annotate("Warning's message does not match", AfterPreprocessing(str, message_matcher)), filename=Annotate("Warning's filname does not match", filename_matcher), lineno=Annotate("Warning's line number does not match", lineno_matcher), line=Annotate("Warning's source line does not match", line_matcher))