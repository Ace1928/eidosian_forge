from __future__ import annotations
def _split_mimebundle(data):
    """Split multi-line string fields in a mimebundle (in-place)"""
    for key, value in list(data.items()):
        if isinstance(value, str) and (key.startswith('text/') or key in _non_text_split_mimes):
            data[key] = value.splitlines(True)
    return data