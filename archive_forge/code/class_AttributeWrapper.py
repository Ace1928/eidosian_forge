import inspect
import sys
from .__wrapt__ import FunctionWrapper
class AttributeWrapper(object):

    def __init__(self, attribute, factory, args, kwargs):
        self.attribute = attribute
        self.factory = factory
        self.args = args
        self.kwargs = kwargs

    def __get__(self, instance, owner):
        value = instance.__dict__[self.attribute]
        return self.factory(value, *self.args, **self.kwargs)

    def __set__(self, instance, value):
        instance.__dict__[self.attribute] = value

    def __delete__(self, instance):
        del instance.__dict__[self.attribute]