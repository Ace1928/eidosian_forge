import warnings
from ._basic import Is
from ._const import Always
from ._datastructures import MatchesListwise, MatchesStructure
from ._higherorder import (
from ._impl import Mismatch
def IsDeprecated(message):
    """
    Make a matcher that checks that a callable produces exactly one
    `DeprecationWarning`.

    :param message: Matcher for the warning message.
    """
    return Warnings(MatchesListwise([WarningMessage(category_type=DeprecationWarning, message=message)]))