import paste.util.threadinglocal as threadinglocal
class StackedObjectRestorer(object):
    """Track StackedObjectProxies and their proxied objects for automatic
    restoration within EvalException's interactive debugger.

    An instance of this class tracks all StackedObjectProxy state in existence
    when unexpected exceptions are raised by WSGI applications housed by
    EvalException and RegistryManager. Like EvalException, this information is
    stored for the life of the process.

    When an unexpected exception occurs and EvalException is present in the
    WSGI stack, save_registry_state is intended to be called to store the
    Registry state and enable automatic restoration on all currently registered
    StackedObjectProxies.

    With restoration enabled, those StackedObjectProxies' _current_obj
    (overwritten by _current_obj_restoration) method's strategy is modified:
    it will return its appropriate proxied object from the restorer when
    a restoration context is active in the current thread.

    The StackedObjectProxies' _push/pop_object methods strategies are also
    changed: they no-op when a restoration context is active in the current
    thread (because the pushing/popping work is all handled by the
    Registry/restorer).

    The request's Registry objects' reglists are restored from the restorer
    when a restoration context begins, enabling the Registry methods to work
    while their changes are tracked by the restorer.

    The overhead of enabling restoration is negligible (another threadlocal
    access for the changed StackedObjectProxy methods) for normal use outside
    of a restoration context, but worth mentioning when combined with
    StackedObjectProxies normal overhead. Once enabled it does not turn off,
    however:

    o Enabling restoration only occurs after an unexpected exception is
    detected. The server is likely to be restarted shortly after the exception
    is raised to fix the cause

    o StackedObjectRestorer is only enabled when EvalException is enabled (not
    on a production server) and RegistryManager exists in the middleware
    stack"""

    def __init__(self):
        self.saved_registry_states = {}
        self.restoration_context_id = threadinglocal.local()

    def save_registry_state(self, environ):
        """Save the state of this request's Registry (if it hasn't already been
        saved) to the saved_registry_states dict, keyed by the request's unique
        identifier"""
        registry = environ.get('paste.registry')
        if not registry or not len(registry.reglist) or self.get_request_id(environ) in self.saved_registry_states:
            return
        self.saved_registry_states[self.get_request_id(environ)] = (registry, registry.reglist[:])
        for reglist in registry.reglist:
            for stacked, obj in reglist.values():
                self.enable_restoration(stacked)

    def get_saved_proxied_obj(self, stacked, request_id):
        """Retrieve the saved object proxied by the specified
        StackedObjectProxy for the request identified by request_id"""
        reglist = self.saved_registry_states[request_id][1]
        stack_level = len(reglist) - 1
        stacked_id = id(stacked)
        while True:
            if stack_level < 0:
                return stacked._current_obj_orig()
            context = reglist[stack_level]
            if stacked_id in context:
                break
            stack_level -= 1
        return context[stacked_id][1]

    def enable_restoration(self, stacked):
        """Replace the specified StackedObjectProxy's methods with their
        respective restoration versions.

        _current_obj_restoration forces recovery of the saved proxied object
        when a restoration context is active in the current thread.

        _push/pop_object_restoration avoid pushing/popping data
        (pushing/popping is only done at the Registry level) when a restoration
        context is active in the current thread"""
        if '_current_obj_orig' in stacked.__dict__:
            return
        for func_name in ('_current_obj', '_push_object', '_pop_object'):
            orig_func = getattr(stacked, func_name)
            restoration_func = getattr(stacked, func_name + '_restoration')
            stacked.__dict__[func_name + '_orig'] = orig_func
            stacked.__dict__[func_name] = restoration_func

    def get_request_id(self, environ):
        """Return a unique identifier for the current request"""
        from paste.evalexception.middleware import get_debug_count
        return get_debug_count(environ)

    def restoration_begin(self, request_id):
        """Enable a restoration context in the current thread for the specified
        request_id"""
        if request_id in self.saved_registry_states:
            registry, reglist = self.saved_registry_states[request_id]
            registry.reglist = reglist
        self.restoration_context_id.request_id = request_id

    def restoration_end(self):
        """Register a restoration context as finished, if one exists"""
        try:
            del self.restoration_context_id.request_id
        except AttributeError:
            pass

    def in_restoration(self):
        """Determine if a restoration context is active for the current thread.
        Returns the request_id it's active for if so, otherwise False"""
        return getattr(self.restoration_context_id, 'request_id', False)