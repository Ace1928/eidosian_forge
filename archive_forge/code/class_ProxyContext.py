class ProxyContext(object):
    __slots__ = ['_obj']

    def __init__(self, obj):
        object.__init__(self)
        object.__setattr__(self, '_obj', obj)

    def __getattribute__(self, name):
        return getattr(object.__getattribute__(self, '_obj'), name)

    def __delattr__(self, name):
        delattr(object.__getattribute__(self, '_obj'), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, '_obj'), name, value)

    def __bool__(self):
        return bool(object.__getattribute__(self, '_obj'))

    def __str__(self):
        return str(object.__getattribute__(self, '_obj'))

    def __repr__(self):
        return repr(object.__getattribute__(self, '_obj'))