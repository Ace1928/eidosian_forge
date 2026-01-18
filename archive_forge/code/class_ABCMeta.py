class ABCMeta(type):
    """Metaclass for defining Abstract Base Classes (ABCs).

        Use this metaclass to create an ABC.  An ABC can be subclassed
        directly, and then acts as a mix-in class.  You can also register
        unrelated concrete classes (even built-in classes) and unrelated
        ABCs as 'virtual subclasses' -- these and their descendants will
        be considered subclasses of the registering ABC by the built-in
        issubclass() function, but the registering ABC won't show up in
        their MRO (Method Resolution Order) nor will method
        implementations defined by the registering ABC be callable (not
        even via super()).
        """

    def __new__(mcls, name, bases, namespace, /, **kwargs):
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        _abc_init(cls)
        return cls

    def register(cls, subclass):
        """Register a virtual subclass of an ABC.

            Returns the subclass, to allow usage as a class decorator.
            """
        return _abc_register(cls, subclass)

    def __instancecheck__(cls, instance):
        """Override for isinstance(instance, cls)."""
        return _abc_instancecheck(cls, instance)

    def __subclasscheck__(cls, subclass):
        """Override for issubclass(subclass, cls)."""
        return _abc_subclasscheck(cls, subclass)

    def _dump_registry(cls, file=None):
        """Debug helper to print the ABC registry."""
        print(f'Class: {cls.__module__}.{cls.__qualname__}', file=file)
        print(f'Inv. counter: {get_cache_token()}', file=file)
        _abc_registry, _abc_cache, _abc_negative_cache, _abc_negative_cache_version = _get_dump(cls)
        print(f'_abc_registry: {_abc_registry!r}', file=file)
        print(f'_abc_cache: {_abc_cache!r}', file=file)
        print(f'_abc_negative_cache: {_abc_negative_cache!r}', file=file)
        print(f'_abc_negative_cache_version: {_abc_negative_cache_version!r}', file=file)

    def _abc_registry_clear(cls):
        """Clear the registry (for debugging or testing)."""
        _reset_registry(cls)

    def _abc_caches_clear(cls):
        """Clear the caches (for debugging or testing)."""
        _reset_caches(cls)