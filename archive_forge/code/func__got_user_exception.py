import sys
from testtools.testresult import ExtendedToOriginalDecorator
def _got_user_exception(self, exc_info, tb_label='traceback'):
    """Called when user code raises an exception.

        If 'exc_info' is a `MultipleExceptions`, then we recurse into it
        unpacking the errors that it's made up from.

        :param exc_info: A sys.exc_info() tuple for the user error.
        :param tb_label: An optional string label for the error.  If
            not specified, will default to 'traceback'.
        :return: 'exception_caught' if we catch one of the exceptions that
            have handlers in 'handlers', otherwise raise the error.
        """
    if exc_info[0] is MultipleExceptions:
        for sub_exc_info in exc_info[1].args:
            self._got_user_exception(sub_exc_info, tb_label)
        return self.exception_caught
    try:
        e = exc_info[1]
        self.case.onException(exc_info, tb_label=tb_label)
    finally:
        del exc_info
    self._exceptions.append(e)
    return self.exception_caught