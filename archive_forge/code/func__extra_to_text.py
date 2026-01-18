from typing import Dict
import dns.enum
import dns.exception
@classmethod
def _extra_to_text(cls, value, current_text):
    if current_text is None:
        return _registered_by_value.get(value)
    if current_text.find('_') >= 0:
        return current_text.replace('_', '-')
    return current_text