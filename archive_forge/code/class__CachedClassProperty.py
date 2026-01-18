import sys
class _CachedClassProperty(object):
    """Cached class property decorator.

  Transforms a class method into a property whose value is computed once
  and then cached as a normal attribute for the life of the class.  Example
  usage:

  >>> class MyClass(object):
  ...   @cached_classproperty
  ...   def value(cls):
  ...     print("Computing value")
  ...     return '<property of %s>' % cls.__name__
  >>> class MySubclass(MyClass):
  ...   pass
  >>> MyClass.value
  Computing value
  '<property of MyClass>'
  >>> MyClass.value  # uses cached value
  '<property of MyClass>'
  >>> MySubclass.value
  Computing value
  '<property of MySubclass>'

  This decorator is similar to `functools.cached_property`, but it adds a
  property to the class, not to individual instances.
  """

    def __init__(self, func):
        self._func = func
        self._cache = {}

    def __get__(self, obj, objtype):
        if objtype not in self._cache:
            self._cache[objtype] = self._func(objtype)
        return self._cache[objtype]

    def __set__(self, obj, value):
        raise AttributeError('property %s is read-only' % self._func.__name__)

    def __delete__(self, obj):
        raise AttributeError('property %s is read-only' % self._func.__name__)