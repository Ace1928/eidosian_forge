class _methodwrapper(object):

    def __init__(self, func, obj, type):
        self.func = func
        self.obj = obj
        self.type = type

    def __call__(self, *args, **kw):
        assert 'self' not in kw and 'cls' not in kw, "You cannot use 'self' or 'cls' arguments to a classinstancemethod"
        return self.func(*(self.obj, self.type) + args, **kw)

    def __repr__(self):
        if self.obj is None:
            return '<bound class method %s.%s>' % (self.type.__name__, self.func.func_name)
        else:
            return '<bound method %s.%s of %r>' % (self.type.__name__, self.func.func_name, self.obj)