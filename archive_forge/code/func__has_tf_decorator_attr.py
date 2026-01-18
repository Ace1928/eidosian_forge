import inspect
from typing import Dict, Any
def _has_tf_decorator_attr(obj):
    """Checks if object has _tf_decorator attribute.

  This check would work for mocked object as well since it would
  check if returned attribute has the right type.

  Args:
    obj: Python object.
  """
    return hasattr(obj, '_tf_decorator') and isinstance(getattr(obj, '_tf_decorator'), TFDecorator)