class _ClassBoolFromEnv:
    """
    Non-data descriptor that reads a transformed environment variable
    as a boolean, and caches the result in the class.
    """

    def __get__(self, inst, klass):
        import os
        for cls in klass.__mro__:
            my_name = None
            for k in dir(klass):
                if k in cls.__dict__ and cls.__dict__[k] is self:
                    my_name = k
                    break
            if my_name is not None:
                break
        else:
            raise RuntimeError('Unable to find self')
        env_name = 'ZOPE_INTERFACE_' + my_name
        val = os.environ.get(env_name, '') == '1'
        val = _NamedBool(val, my_name)
        setattr(klass, my_name, val)
        setattr(klass, 'ORIG_' + my_name, self)
        return val