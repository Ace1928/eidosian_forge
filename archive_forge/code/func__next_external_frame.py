import sys
def _next_external_frame(frame):
    """Find the next frame that doesn't involve CPython internals."""
    frame = frame.f_back
    while frame is not None and _is_internal_frame(frame):
        frame = frame.f_back
    return frame