import inspect
import types
Like getattr, but with a few extra sanity checks:

    - If obj is a class, ignore everything except class methods
    - Check if obj is a proxy that claims to have all attributes
    - Catch attribute access failing with any exception
    - Check that the attribute is a callable object

    Returns the method or None.
    