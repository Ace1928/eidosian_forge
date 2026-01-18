import inspect
from typing import Dict, Any
def _get_bound_instance(target):
    """Returns the instance any of the targets is attached to."""
    decorators, target = unwrap(target)
    for decorator in decorators:
        if inspect.ismethod(decorator.decorated_target):
            return decorator.decorated_target.__self__