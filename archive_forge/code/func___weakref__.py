import sys
import operator
import inspect
@property
def __weakref__(self):
    return self.__wrapped__.__weakref__