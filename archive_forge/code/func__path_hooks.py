import _imp
import _io
import sys
import _warnings
import marshal
@staticmethod
def _path_hooks(path):
    """Search sys.path_hooks for a finder for 'path'."""
    if sys.path_hooks is not None and (not sys.path_hooks):
        _warnings.warn('sys.path_hooks is empty', ImportWarning)
    for hook in sys.path_hooks:
        try:
            return hook(path)
        except ImportError:
            continue
    else:
        return None