import abc
import enum
import threading  # pylint: disable=unused-import
class NoSuchMethodError(Exception):
    """Indicates that an unrecognized operation has been called.

    Attributes:
      code: A code value to communicate to the other side of the operation
        along with indication of operation termination. May be None.
      details: A details value to communicate to the other side of the
        operation along with indication of operation termination. May be None.
    """

    def __init__(self, code, details):
        """Constructor.

        Args:
          code: A code value to communicate to the other side of the operation
            along with indication of operation termination. May be None.
          details: A details value to communicate to the other side of the
            operation along with indication of operation termination. May be None.
        """
        super(NoSuchMethodError, self).__init__()
        self.code = code
        self.details = details