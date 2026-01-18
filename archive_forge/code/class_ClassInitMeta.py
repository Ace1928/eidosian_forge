class ClassInitMeta(type):

    def __new__(meta, class_name, bases, new_attrs):
        cls = type.__new__(meta, class_name, bases, new_attrs)
        if new_attrs.has_key('__classinit__') and (not isinstance(cls.__classinit__, staticmethod)):
            setattr(cls, '__classinit__', staticmethod(cls.__classinit__.im_func))
        if hasattr(cls, '__classinit__'):
            cls.__classinit__(cls, new_attrs)
        return cls