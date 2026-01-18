import abc
from typing import Any
class StrPresentationProvider(_AbstractProvider):
    """
    Implement this in an extension to provide a str presentation for a type
    """

    def get_str_in_context(self, val: Any, context: str):
        """
        :param val:
            This is the object for which we want a string representation.

        :param context:
            This is the context in which the variable is being requested. Valid values:
                "watch",
                "repl",
                "hover",
                "clipboard"

        :note: this method is not required (if it's not available, get_str is called directly,
               so, it's only needed if the string representation needs to be converted based on
               the context).
        """
        return self.get_str(val)

    @abc.abstractmethod
    def get_str(self, val):
        raise NotImplementedError