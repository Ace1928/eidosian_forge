import abc
from oslo_serialization import jsonutils
import six
class ValidatorDescriptor(object):

    def __init__(self, name, func=None):
        self.name = name
        self.func = func

    def __set__(self, instance, value):
        if value is not None:
            if self.func is not None:
                if self.func(value):
                    instance.__dict__[self.name] = value
                else:
                    raise ValueError('%s failed validation: %s' % (self.name, self.func))
            else:
                instance.__dict__[self.name] = value
        else:
            raise ValueError('%s must not be None.' % self.name)