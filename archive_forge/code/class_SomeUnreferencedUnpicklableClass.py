import dill
import doctest
import logging
class SomeUnreferencedUnpicklableClass(object):

    def __reduce__(self):
        raise Exception