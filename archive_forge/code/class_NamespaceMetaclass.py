import sys, re
class NamespaceMetaclass(type):

    def __getattr__(self, name):
        if name[:1] == '_':
            raise AttributeError(name)
        if self == Namespace:
            raise ValueError('Namespace class is abstract')
        tagspec = self.__tagspec__
        if tagspec is not None and name not in tagspec:
            raise AttributeError(name)
        classattr = {}
        if self.__stickyname__:
            classattr['xmlname'] = name
        cls = type(name, (self.__tagclass__,), classattr)
        setattr(self, name, cls)
        return cls