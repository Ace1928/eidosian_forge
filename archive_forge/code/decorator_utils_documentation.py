import sys
Cached class property decorator.

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
  