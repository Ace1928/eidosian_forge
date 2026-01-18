import inspect
import weakref
A function cache based on unbound function objects.

  Using the function for the cache key allows efficient handling of object
  methods.

  Unlike the _CodeObjectCache, this discriminates between different functions
  even if they have the same code. This is needed for decorators that may
  masquerade as another function.
  