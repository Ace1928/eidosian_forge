from __future__ import annotations
import sys
import traceback
def _safe_str(s, errors='replace', file=None):
    if isinstance(s, str):
        return s
    try:
        return str(s)
    except Exception as exc:
        return '<Unrepresentable {!r}: {!r} {!r}>'.format(type(s), exc, '\n'.join(traceback.format_stack()))