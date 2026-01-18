from eventlet.event import Event
from eventlet import greenthread
import collections
class DAGPool:
    """
    A DAGPool is a pool that constrains greenthreads, not by max concurrency,
    but by data dependencies.

    This is a way to implement general DAG dependencies. A simple dependency
    tree (flowing in either direction) can straightforwardly be implemented
    using recursion and (e.g.)
    :meth:`GreenThread.imap() <eventlet.greenthread.GreenThread.imap>`.
    What gets complicated is when a given node depends on several other nodes
    as well as contributing to several other nodes.

    With DAGPool, you concurrently launch all applicable greenthreads; each
    will proceed as soon as it has all required inputs. The DAG is implicit in
    which items are required by each greenthread.

    Each greenthread is launched in a DAGPool with a key: any value that can
    serve as a Python dict key. The caller also specifies an iterable of other
    keys on which this greenthread depends. This iterable may be empty.

    The greenthread callable must accept (key, results), where:

    key
        is its own key

    results
        is an iterable of (key, value) pairs.

    A newly-launched DAGPool greenthread is entered immediately, and can
    perform any necessary setup work. At some point it will iterate over the
    (key, value) pairs from the passed 'results' iterable. Doing so blocks the
    greenthread until a value is available for each of the keys specified in
    its initial dependencies iterable. These (key, value) pairs are delivered
    in chronological order, *not* the order in which they are initially
    specified: each value will be delivered as soon as it becomes available.

    The value returned by a DAGPool greenthread becomes the value for its
    key, which unblocks any other greenthreads waiting on that key.

    If a DAGPool greenthread terminates with an exception instead of returning
    a value, attempting to retrieve the value raises :class:`PropagateError`,
    which binds the key of the original greenthread and the original
    exception. Unless the greenthread attempting to retrieve the value handles
    PropagateError, that exception will in turn be wrapped in a PropagateError
    of its own, and so forth. The code that ultimately handles PropagateError
    can follow the chain of PropagateError.exc attributes to discover the flow
    of that exception through the DAG of greenthreads.

    External greenthreads may also interact with a DAGPool. See :meth:`wait_each`,
    :meth:`waitall`, :meth:`post`.

    It is not recommended to constrain external DAGPool producer greenthreads
    in a :class:`GreenPool <eventlet.greenpool.GreenPool>`: it may be hard to
    provably avoid deadlock.

    .. automethod:: __init__
    .. automethod:: __getitem__
    """
    _Coro = collections.namedtuple('_Coro', ('greenthread', 'pending'))

    def __init__(self, preload={}):
        """
        DAGPool can be prepopulated with an initial dict or iterable of (key,
        value) pairs. These (key, value) pairs are of course immediately
        available for any greenthread that depends on any of those keys.
        """
        try:
            iteritems = preload.items()
        except AttributeError:
            iteritems = preload
        self.values = dict(iteritems)
        self.coros = {}
        self.event = Event()

    def waitall(self):
        """
        waitall() blocks the calling greenthread until there is a value for
        every DAGPool greenthread launched by :meth:`spawn`. It returns a dict
        containing all :class:`preload data <DAGPool>`, all data from
        :meth:`post` and all values returned by spawned greenthreads.

        See also :meth:`wait`.
        """
        return self.wait()

    def wait(self, keys=_MISSING):
        """
        *keys* is an optional iterable of keys. If you omit the argument, it
        waits for all the keys from :class:`preload data <DAGPool>`, from
        :meth:`post` calls and from :meth:`spawn` calls: in other words, all
        the keys of which this DAGPool is aware.

        wait() blocks the calling greenthread until all of the relevant keys
        have values. wait() returns a dict whose keys are the relevant keys,
        and whose values come from the *preload* data, from values returned by
        DAGPool greenthreads or from :meth:`post` calls.

        If a DAGPool greenthread terminates with an exception, wait() will
        raise :class:`PropagateError` wrapping that exception. If more than
        one greenthread terminates with an exception, it is indeterminate
        which one wait() will raise.

        If an external greenthread posts a :class:`PropagateError` instance,
        wait() will raise that PropagateError. If more than one greenthread
        posts PropagateError, it is indeterminate which one wait() will raise.

        See also :meth:`wait_each_success`, :meth:`wait_each_exception`.
        """
        return dict(self.wait_each(keys))

    def wait_each(self, keys=_MISSING):
        """
        *keys* is an optional iterable of keys. If you omit the argument, it
        waits for all the keys from :class:`preload data <DAGPool>`, from
        :meth:`post` calls and from :meth:`spawn` calls: in other words, all
        the keys of which this DAGPool is aware.

        wait_each() is a generator producing (key, value) pairs as a value
        becomes available for each requested key. wait_each() blocks the
        calling greenthread until the next value becomes available. If the
        DAGPool was prepopulated with values for any of the relevant keys, of
        course those can be delivered immediately without waiting.

        Delivery order is intentionally decoupled from the initial sequence of
        keys: each value is delivered as soon as it becomes available. If
        multiple keys are available at the same time, wait_each() delivers
        each of the ready ones in arbitrary order before blocking again.

        The DAGPool does not distinguish between a value returned by one of
        its own greenthreads and one provided by a :meth:`post` call or *preload* data.

        The wait_each() generator terminates (raises StopIteration) when all
        specified keys have been delivered. Thus, typical usage might be:

        ::

            for key, value in dagpool.wait_each(keys):
                # process this ready key and value
            # continue processing now that we've gotten values for all keys

        By implication, if you pass wait_each() an empty iterable of keys, it
        returns immediately without yielding anything.

        If the value to be delivered is a :class:`PropagateError` exception object, the
        generator raises that PropagateError instead of yielding it.

        See also :meth:`wait_each_success`, :meth:`wait_each_exception`.
        """
        return self._wait_each(self._get_keyset_for_wait_each(keys))

    def wait_each_success(self, keys=_MISSING):
        """
        wait_each_success() filters results so that only success values are
        yielded. In other words, unlike :meth:`wait_each`, wait_each_success()
        will not raise :class:`PropagateError`. Not every provided (or
        defaulted) key will necessarily be represented, though naturally the
        generator will not finish until all have completed.

        In all other respects, wait_each_success() behaves like :meth:`wait_each`.
        """
        for key, value in self._wait_each_raw(self._get_keyset_for_wait_each(keys)):
            if not isinstance(value, PropagateError):
                yield (key, value)

    def wait_each_exception(self, keys=_MISSING):
        """
        wait_each_exception() filters results so that only exceptions are
        yielded. Not every provided (or defaulted) key will necessarily be
        represented, though naturally the generator will not finish until
        all have completed.

        Unlike other DAGPool methods, wait_each_exception() simply yields
        :class:`PropagateError` instances as values rather than raising them.

        In all other respects, wait_each_exception() behaves like :meth:`wait_each`.
        """
        for key, value in self._wait_each_raw(self._get_keyset_for_wait_each(keys)):
            if isinstance(value, PropagateError):
                yield (key, value)

    def _get_keyset_for_wait_each(self, keys):
        """
        wait_each(), wait_each_success() and wait_each_exception() promise
        that if you pass an iterable of keys, the method will wait for results
        from those keys -- but if you omit the keys argument, the method will
        wait for results from all known keys. This helper implements that
        distinction, returning a set() of the relevant keys.
        """
        if keys is not _MISSING:
            return set(keys)
        else:
            return set(self.coros.keys()) | set(self.values.keys())

    def _wait_each(self, pending):
        """
        When _wait_each() encounters a value of PropagateError, it raises it.

        In all other respects, _wait_each() behaves like _wait_each_raw().
        """
        for key, value in self._wait_each_raw(pending):
            yield (key, self._value_or_raise(value))

    @staticmethod
    def _value_or_raise(value):
        if isinstance(value, PropagateError):
            raise value
        return value

    def _wait_each_raw(self, pending):
        """
        pending is a set() of keys for which we intend to wait. THIS SET WILL
        BE DESTRUCTIVELY MODIFIED: as each key acquires a value, that key will
        be removed from the passed 'pending' set.

        _wait_each_raw() does not treat a PropagateError instance specially:
        it will be yielded to the caller like any other value.

        In all other respects, _wait_each_raw() behaves like wait_each().
        """
        while True:
            for key in pending.copy():
                value = self.values.get(key, _MISSING)
                if value is not _MISSING:
                    pending.remove(key)
                    yield (key, value)
            if not pending:
                break
            self.event.wait()

    def spawn(self, key, depends, function, *args, **kwds):
        """
        Launch the passed *function(key, results, ...)* as a greenthread,
        passing it:

        - the specified *key*
        - an iterable of (key, value) pairs
        - whatever other positional args or keywords you specify.

        Iterating over the *results* iterable behaves like calling
        :meth:`wait_each(depends) <DAGPool.wait_each>`.

        Returning from *function()* behaves like
        :meth:`post(key, return_value) <DAGPool.post>`.

        If *function()* terminates with an exception, that exception is wrapped
        in :class:`PropagateError` with the greenthread's *key* and (effectively) posted
        as the value for that key. Attempting to retrieve that value will
        raise that PropagateError.

        Thus, if the greenthread with key 'a' terminates with an exception,
        and greenthread 'b' depends on 'a', when greenthread 'b' attempts to
        iterate through its *results* argument, it will encounter
        PropagateError. So by default, an uncaught exception will propagate
        through all the downstream dependencies.

        If you pass :meth:`spawn` a key already passed to spawn() or :meth:`post`, spawn()
        raises :class:`Collision`.
        """
        if key in self.coros or key in self.values:
            raise Collision(key)
        pending = set(depends)
        newcoro = greenthread.spawn(self._wrapper, function, key, self._wait_each(pending), *args, **kwds)
        self.coros[key] = self._Coro(newcoro, pending)

    def _wrapper(self, function, key, results, *args, **kwds):
        """
        This wrapper runs the top-level function in a DAGPool greenthread,
        posting its return value (or PropagateError) to the DAGPool.
        """
        try:
            result = function(key, results, *args, **kwds)
        except Exception as err:
            result = PropagateError(key, err)
        finally:
            del self.coros[key]
        try:
            self.post(key, result)
        except Collision:
            pass
        return result

    def spawn_many(self, depends, function, *args, **kwds):
        """
        spawn_many() accepts a single *function* whose parameters are the same
        as for :meth:`spawn`.

        The difference is that spawn_many() accepts a dependency dict
        *depends*. A new greenthread is spawned for each key in the dict. That
        dict key's value should be an iterable of other keys on which this
        greenthread depends.

        If the *depends* dict contains any key already passed to :meth:`spawn`
        or :meth:`post`, spawn_many() raises :class:`Collision`. It is
        indeterminate how many of the other keys in *depends* will have
        successfully spawned greenthreads.
        """
        for key, deps in depends.items():
            self.spawn(key, deps, function, *args, **kwds)

    def kill(self, key):
        """
        Kill the greenthread that was spawned with the specified *key*.

        If no such greenthread was spawned, raise KeyError.
        """
        self.coros[key].greenthread.kill()
        del self.coros[key]

    def post(self, key, value, replace=False):
        """
        post(key, value) stores the passed *value* for the passed *key*. It
        then causes each greenthread blocked on its results iterable, or on
        :meth:`wait_each(keys) <DAGPool.wait_each>`, to check for new values.
        A waiting greenthread might not literally resume on every single
        post() of a relevant key, but the first post() of a relevant key
        ensures that it will resume eventually, and when it does it will catch
        up with all relevant post() calls.

        Calling post(key, value) when there is a running greenthread with that
        same *key* raises :class:`Collision`. If you must post(key, value) instead of
        letting the greenthread run to completion, you must first call
        :meth:`kill(key) <DAGPool.kill>`.

        The DAGPool implicitly post()s the return value from each of its
        greenthreads. But a greenthread may explicitly post() a value for its
        own key, which will cause its return value to be discarded.

        Calling post(key, value, replace=False) (the default *replace*) when a
        value for that key has already been posted, by any means, raises
        :class:`Collision`.

        Calling post(key, value, replace=True) when a value for that key has
        already been posted, by any means, replaces the previously-stored
        value. However, that may make it complicated to reason about the
        behavior of greenthreads waiting on that key.

        After a post(key, value1) followed by post(key, value2, replace=True),
        it is unspecified which pending :meth:`wait_each([key...]) <DAGPool.wait_each>`
        calls (or greenthreads iterating over *results* involving that key)
        will observe *value1* versus *value2*. It is guaranteed that
        subsequent wait_each([key...]) calls (or greenthreads spawned after
        that point) will observe *value2*.

        A successful call to
        post(key, :class:`PropagateError(key, ExceptionSubclass) <PropagateError>`)
        ensures that any subsequent attempt to retrieve that key's value will
        raise that PropagateError instance.
        """
        coro = self.coros.get(key, _MISSING)
        if coro is not _MISSING and coro.greenthread is not greenthread.getcurrent():
            raise Collision(key)
        if key in self.values and (not replace):
            raise Collision(key)
        self.values[key] = value
        self.event.send()
        self.event = Event()

    def __getitem__(self, key):
        """
        __getitem__(key) (aka dagpool[key]) blocks until *key* has a value,
        then delivers that value.
        """
        for _, value in self.wait_each((key,)):
            return value

    def get(self, key, default=None):
        """
        get() returns the value for *key*. If *key* does not yet have a value,
        get() returns *default*.
        """
        return self._value_or_raise(self.values.get(key, default))

    def keys(self):
        """
        Return a snapshot tuple of keys for which we currently have values.
        """
        return tuple(self.values.keys())

    def items(self):
        """
        Return a snapshot tuple of currently-available (key, value) pairs.
        """
        return tuple(((key, self._value_or_raise(value)) for key, value in self.values.items()))

    def running(self):
        """
        Return number of running DAGPool greenthreads. This includes
        greenthreads blocked while iterating through their *results* iterable,
        that is, greenthreads waiting on values from other keys.
        """
        return len(self.coros)

    def running_keys(self):
        """
        Return keys for running DAGPool greenthreads. This includes
        greenthreads blocked while iterating through their *results* iterable,
        that is, greenthreads waiting on values from other keys.
        """
        return tuple(self.coros.keys())

    def waiting(self):
        """
        Return number of waiting DAGPool greenthreads, that is, greenthreads
        still waiting on values from other keys. This explicitly does *not*
        include external greenthreads waiting on :meth:`wait`,
        :meth:`waitall`, :meth:`wait_each`.
        """
        return len(self.waiting_for())

    def waiting_for(self, key=_MISSING):
        """
        waiting_for(key) returns a set() of the keys for which the DAGPool
        greenthread spawned with that *key* is still waiting. If you pass a
        *key* for which no greenthread was spawned, waiting_for() raises
        KeyError.

        waiting_for() without argument returns a dict. Its keys are the keys
        of DAGPool greenthreads still waiting on one or more values. In the
        returned dict, the value of each such key is the set of other keys for
        which that greenthread is still waiting.

        This method allows diagnosing a "hung" DAGPool. If certain
        greenthreads are making no progress, it's possible that they are
        waiting on keys for which there is no greenthread and no :meth:`post` data.
        """
        available = set(self.values.keys())
        if key is not _MISSING:
            coro = self.coros.get(key, _MISSING)
            if coro is _MISSING:
                self.values[key]
                return set()
            else:
                return coro.pending - available
        return {key: pending for key, pending in ((key, coro.pending - available) for key, coro in self.coros.items()) if pending}