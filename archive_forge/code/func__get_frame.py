from pathlib import Path
import sys
import inspect
import warnings
def _get_frame(level):
    """Get the frame at the given stack level."""
    if hasattr(sys, '_getframe'):
        frame = sys._getframe(level + 1)
    else:
        frame = inspect.stack(context=0)[level + 1].frame
    return frame