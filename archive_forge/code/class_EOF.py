import traceback
import sys
class EOF(ExceptionPexpect):
    """Raised when EOF is read from a child.
    This usually means the child has exited."""