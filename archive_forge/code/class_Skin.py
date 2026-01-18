class Skin(object):
    """
    The meta-programming I{skin} around the L{Properties} object.
    @ivar __pts__: The wrapped object.
    @type __pts__: L{Properties}.
    """

    def __init__(self, domain, definitions, kwargs):
        self.__pts__ = Properties(domain, definitions, kwargs)

    def __setattr__(self, name, value):
        builtin = name.startswith('__') and name.endswith('__')
        if builtin:
            self.__dict__[name] = value
            return
        self.__pts__.set(name, value)

    def __getattr__(self, name):
        return self.__pts__.get(name)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.__pts__)