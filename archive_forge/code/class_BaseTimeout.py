import functools
import logging
import sys
class BaseTimeout(object):
    """Context manager for limiting in the time the execution of a block

    :param seconds: ``float`` or ``int`` duration enabled to run the context
      manager block
    :param swallow_exc: ``False`` if you want to manage the
      ``TimeoutException`` (or any other) in an outer ``try ... except``
      structure. ``True`` (default) if you just want to check the execution of
      the block with the ``state`` attribute of the context manager.
    """
    EXECUTED, EXECUTING, TIMED_OUT, INTERRUPTED, CANCELED = range(5)

    def __init__(self, seconds, swallow_exc=True):
        self.seconds = seconds
        self.swallow_exc = swallow_exc
        self.state = BaseTimeout.EXECUTED

    def __bool__(self):
        return self.state in (BaseTimeout.EXECUTED, BaseTimeout.EXECUTING, BaseTimeout.CANCELED)
    __nonzero__ = __bool__

    def __repr__(self):
        """Debug helper
        """
        return '<{0} in state: {1}>'.format(self.__class__.__name__, self.state)

    def __enter__(self):
        self.state = BaseTimeout.EXECUTING
        self.setup_interrupt()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is TimeoutException:
            if self.state != BaseTimeout.TIMED_OUT:
                self.state = BaseTimeout.INTERRUPTED
                self.suppress_interrupt()
            LOG.warning('Code block execution exceeded {0} seconds timeout'.format(self.seconds), exc_info=(exc_type, exc_val, exc_tb))
            return self.swallow_exc
        else:
            if exc_type is None:
                self.state = BaseTimeout.EXECUTED
            self.suppress_interrupt()
        return False

    def cancel(self):
        """In case in the block you realize you don't need anymore
       limitation"""
        self.state = BaseTimeout.CANCELED
        self.suppress_interrupt()

    def suppress_interrupt(self):
        """Removes/neutralizes the feature that interrupts the executed block
        """
        raise NotImplementedError

    def setup_interrupt(self):
        """Installs/initializes the feature that interrupts the executed block
        """
        raise NotImplementedError