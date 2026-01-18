import sys
class CallMany(object):
    """A stack of functions which will all be called on __call__.

    CallMany also acts as a context manager for convenience.

    Functions are called in last pushed first executed order.

    This is used by Fixture to manage its addCleanup feature.
    """

    def __init__(self):
        self._cleanups = []

    def push(self, cleanup, *args, **kwargs):
        """Add a function to be called from __call__.

        On __call__ all functions are called - see __call__ for details on how
        multiple exceptions are handled.

        :param cleanup: A callable to call during cleanUp.
        :param *args: Positional args for cleanup.
        :param kwargs: Keyword args for cleanup.
        :return: None
        """
        self._cleanups.append((cleanup, args, kwargs))

    def __call__(self, raise_errors=True):
        """Run all the registered functions.

        :param raise_errors: Deprecated parameter from before testtools gained
            MultipleExceptions. raise_errors defaults to True. When True
            if exception(s) are raised while running functions, they are
            re-raised after all the functions have run.  If multiple exceptions
            are raised, they are all wrapped into a MultipleExceptions object,
            and that is raised.

            Thus, to catch a specific exception from a function run by __call__,
            you need to catch both the exception and MultipleExceptions, and
            then check within a MultipleExceptions instance for an occurrence of
            the type you wish to catch.
        :return: Either None or a list of the exc_info() for each exception
            that occurred if raise_errors was False.
        """
        cleanups = reversed(self._cleanups)
        self._cleanups = []
        result = []
        for cleanup, args, kwargs in cleanups:
            try:
                cleanup(*args, **kwargs)
            except Exception:
                result.append(sys.exc_info())
        if result and raise_errors:
            if 1 == len(result):
                error = result[0]
                raise error[1].with_traceback(error[2])
            else:
                raise MultipleExceptions(*result)
        if not raise_errors:
            return result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self()
        return False