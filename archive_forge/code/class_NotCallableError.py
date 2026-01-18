from __future__ import annotations
from typing import ClassVar
class NotCallableError(TypeError):
    """
    A field requiring a callable has been set with a value that is not
    callable.

    .. versionadded:: 19.2.0
    """

    def __init__(self, msg, value):
        super(TypeError, self).__init__(msg, value)
        self.msg = msg
        self.value = value

    def __str__(self):
        return str(self.msg)