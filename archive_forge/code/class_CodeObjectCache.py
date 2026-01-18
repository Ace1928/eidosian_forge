import inspect
import weakref
class CodeObjectCache(_TransformedFnCache):
    """A function cache based on code objects.

  Code objects are good proxies for the source code of a function.

  This cache efficiently handles functions that share code objects, such as
  functions defined in a loop, bound methods, etc.

  The cache falls back to the function object, if it doesn't have a code object.
  """

    def _get_key(self, entity):
        if hasattr(entity, '__code__'):
            return entity.__code__
        else:
            return entity