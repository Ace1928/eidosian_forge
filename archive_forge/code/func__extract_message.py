import json
import re
import typing as ty
from requests import exceptions as _rex
def _extract_message(obj):
    if isinstance(obj, dict):
        if obj.get('message'):
            return obj['message']
        elif obj.get('faultstring'):
            return obj['faultstring']
    elif isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            pass
        else:
            return _extract_message(obj)