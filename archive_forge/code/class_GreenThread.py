from collections import deque
import sys
from greenlet import GreenletExit
from eventlet import event
from eventlet import hubs
from eventlet import support
from eventlet import timeout
from eventlet.hubs import timer
from eventlet.support import greenlets as greenlet
import warnings
class GreenThread(greenlet.greenlet):
    """The GreenThread class is a type of Greenlet which has the additional
    property of being able to retrieve the return value of the main function.
    Do not construct GreenThread objects directly; call :func:`spawn` to get one.
    """

    def __init__(self, parent):
        greenlet.greenlet.__init__(self, self.main, parent)
        self._exit_event = event.Event()
        self._resolving_links = False
        self._exit_funcs = None

    def __await__(self):
        """
        Enable ``GreenThread``s to be ``await``ed in ``async`` functions.
        """
        from eventlet.hubs.asyncio import Hub
        hub = hubs.get_hub()
        if not isinstance(hub, Hub):
            raise RuntimeError("This API only works with eventlet's asyncio hub. " + 'To use it, set an EVENTLET_HUB=asyncio environment variable.')
        future = hub.loop.create_future()

        def got_future_result(future):
            if future.cancelled() and (not self.dead):
                self.kill()
        future.add_done_callback(got_future_result)

        def got_gthread_result(gthread):
            if future.done():
                return
            try:
                result = gthread.wait()
                future.set_result(result)
            except GreenletExit:
                future.cancel()
            except BaseException as e:
                future.set_exception(e)
        self.link(got_gthread_result)
        return future.__await__()

    def wait(self):
        """ Returns the result of the main function of this GreenThread.  If the
        result is a normal return value, :meth:`wait` returns it.  If it raised
        an exception, :meth:`wait` will raise the same exception (though the
        stack trace will unavoidably contain some frames from within the
        greenthread module)."""
        return self._exit_event.wait()

    def link(self, func, *curried_args, **curried_kwargs):
        """ Set up a function to be called with the results of the GreenThread.

        The function must have the following signature::

            def func(gt, [curried args/kwargs]):

        When the GreenThread finishes its run, it calls *func* with itself
        and with the `curried arguments <http://en.wikipedia.org/wiki/Currying>`_ supplied
        at link-time.  If the function wants to retrieve the result of the GreenThread,
        it should call wait() on its first argument.

        Note that *func* is called within execution context of
        the GreenThread, so it is possible to interfere with other linked
        functions by doing things like switching explicitly to another
        greenthread.
        """
        if self._exit_funcs is None:
            self._exit_funcs = deque()
        self._exit_funcs.append((func, curried_args, curried_kwargs))
        if self._exit_event.ready():
            self._resolve_links()

    def unlink(self, func, *curried_args, **curried_kwargs):
        """ remove linked function set by :meth:`link`

        Remove successfully return True, otherwise False
        """
        if not self._exit_funcs:
            return False
        try:
            self._exit_funcs.remove((func, curried_args, curried_kwargs))
            return True
        except ValueError:
            return False

    def main(self, function, args, kwargs):
        try:
            result = function(*args, **kwargs)
        except:
            self._exit_event.send_exception(*sys.exc_info())
            self._resolve_links()
            raise
        else:
            self._exit_event.send(result)
            self._resolve_links()

    def _resolve_links(self):
        if self._resolving_links:
            return
        if not self._exit_funcs:
            return
        self._resolving_links = True
        try:
            while self._exit_funcs:
                f, ca, ckw = self._exit_funcs.popleft()
                f(self, *ca, **ckw)
        finally:
            self._resolving_links = False

    def kill(self, *throw_args):
        """Kills the greenthread using :func:`kill`.  After being killed
        all calls to :meth:`wait` will raise *throw_args* (which default
        to :class:`greenlet.GreenletExit`)."""
        return kill(self, *throw_args)

    def cancel(self, *throw_args):
        """Kills the greenthread using :func:`kill`, but only if it hasn't
        already started running.  After being canceled,
        all calls to :meth:`wait` will raise *throw_args* (which default
        to :class:`greenlet.GreenletExit`)."""
        return cancel(self, *throw_args)