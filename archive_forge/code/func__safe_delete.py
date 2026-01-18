import functools
import types
from fixtures import Fixture
def _safe_delete(self, obj, attribute):
    """Delete obj.attribute handling the case where its missing."""
    sentinel = object()
    if getattr(obj, attribute, sentinel) is not sentinel:
        delattr(obj, attribute)