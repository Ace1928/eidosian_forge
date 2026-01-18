import itertools
import sys
from fixtures.callmany import (
class FunctionFixture(Fixture):
    """An adapter to use function(s) as a Fixture.

    Typically used when an existing object or function interface exists but you
    wish to use it as a Fixture (e.g. because fixtures are in use in your test
    suite and this will fit in better).

    To adapt an object with differently named setUp and cleanUp methods:
    fixture = FunctionFixture(object.install, object.__class__.remove)
    Note that the indirection via __class__ is to get an unbound method
    which can accept the result from install. See also MethodFixture which
    is specialised for objects.

    To adapt functions:
    fixture = FunctionFixture(tempfile.mkdtemp, shutil.rmtree)

    With a reset function:
    fixture = FunctionFixture(setup, cleanup, reset)

    :ivar fn_result: The result of the setup_fn. Undefined outside of the
        setUp, cleanUp context.
    """

    def __init__(self, setup_fn, cleanup_fn=None, reset_fn=None):
        """Create a FunctionFixture.

        :param setup_fn: A callable which takes no parameters and returns the
            thing you want to use. e.g.
            def setup_fn():
                return 42
            The result of setup_fn is assigned to the fn_result attribute bu
            FunctionFixture.setUp.
        :param cleanup_fn: Optional callable which takes a single parameter, which
            must be that which is returned from the setup_fn. This is called
            from cleanUp.
        :param reset_fn: Optional callable which takes a single parameter like
            cleanup_fn, but also returns a new object for use as the fn_result:
            if defined this replaces the use of cleanup_fn and setup_fn when
            reset() is called.
        """
        super(FunctionFixture, self).__init__()
        self.setup_fn = setup_fn
        self.cleanup_fn = cleanup_fn
        self.reset_fn = reset_fn

    def _setUp(self):
        fn_result = self.setup_fn()
        self._maybe_cleanup(fn_result)

    def reset(self):
        if self.reset_fn is None:
            super(FunctionFixture, self).reset()
        else:
            self._clear_cleanups()
            fn_result = self.reset_fn(self.fn_result)
            self._maybe_cleanup(fn_result)

    def _maybe_cleanup(self, fn_result):
        self.addCleanup(delattr, self, 'fn_result')
        if self.cleanup_fn is not None:
            self.addCleanup(self.cleanup_fn, fn_result)
        self.fn_result = fn_result