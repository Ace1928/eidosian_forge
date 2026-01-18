from __future__ import annotations
def _is_json_mime(mime):
    """Is a key a JSON mime-type that should be left alone?"""
    return mime == 'application/json' or (mime.startswith('application/') and mime.endswith('+json'))