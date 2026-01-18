import logging
from .enabled import EnabledExtensionManager
from .exception import NoMatches
class DispatchExtensionManager(EnabledExtensionManager):
    """Loads all plugins and filters on execution.

    This is useful for long-running processes that need to pass
    different inputs to different extensions.

    :param namespace: The namespace for the entry points.
    :type namespace: str
    :param check_func: Function to determine which extensions to load.
    :type check_func: callable
    :param invoke_on_load: Boolean controlling whether to invoke the
        object returned by the entry point after the driver is loaded.
    :type invoke_on_load: bool
    :param invoke_args: Positional arguments to pass when invoking
        the object returned by the entry point. Only used if invoke_on_load
        is True.
    :type invoke_args: tuple
    :param invoke_kwds: Named arguments to pass when invoking
        the object returned by the entry point. Only used if invoke_on_load
        is True.
    :type invoke_kwds: dict
    :param propagate_map_exceptions: Boolean controlling whether exceptions
        are propagated up through the map call or whether they are logged and
        then ignored
    :type invoke_on_load: bool
    """

    def map(self, filter_func, func, *args, **kwds):
        """Iterate over the extensions invoking func() for any where
        filter_func() returns True.

        The signature of filter_func() should be::

            def filter_func(ext, *args, **kwds):
                pass

        The first argument to filter_func(), 'ext', is the
        :class:`~stevedore.extension.Extension`
        instance. filter_func() should return True if the extension
        should be invoked for the input arguments.

        The signature for func() should be::

            def func(ext, *args, **kwds):
                pass

        The first argument to func(), 'ext', is the
        :class:`~stevedore.extension.Extension` instance.

        Exceptions raised from within func() are propagated up and
        processing stopped if self.propagate_map_exceptions is True,
        otherwise they are logged and ignored.

        :param filter_func: Callable to test each extension.
        :param func: Callable to invoke for each extension.
        :param args: Variable arguments to pass to func()
        :param kwds: Keyword arguments to pass to func()
        :returns: List of values returned from func()
        """
        if not self.extensions:
            raise NoMatches('No %s extensions found' % self.namespace)
        response = []
        for e in self.extensions:
            if filter_func(e, *args, **kwds):
                self._invoke_one_plugin(response.append, func, e, args, kwds)
        return response

    def map_method(self, filter_func, method_name, *args, **kwds):
        """Iterate over the extensions invoking each one's object method called
        `method_name` for any where filter_func() returns True.

        This is equivalent of using :meth:`map` with func set to
        `lambda x: x.obj.method_name()`
        while being more convenient.

        Exceptions raised from within the called method are propagated up
        and processing stopped if self.propagate_map_exceptions is True,
        otherwise they are logged and ignored.

        .. versionadded:: 0.12

        :param filter_func: Callable to test each extension.
        :param method_name: The extension method name to call
                            for each extension.
        :param args: Variable arguments to pass to method
        :param kwds: Keyword arguments to pass to method
        :returns: List of values returned from methods
        """
        return self.map(filter_func, self._call_extension_method, method_name, *args, **kwds)