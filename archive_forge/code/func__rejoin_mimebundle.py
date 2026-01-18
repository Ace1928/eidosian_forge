from __future__ import annotations
def _rejoin_mimebundle(data):
    """Rejoin the multi-line string fields in a mimebundle (in-place)"""
    for key, value in list(data.items()):
        if not _is_json_mime(key) and isinstance(value, list) and all((isinstance(line, str) for line in value)):
            data[key] = ''.join(value)
    return data