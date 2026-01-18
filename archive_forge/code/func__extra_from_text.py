from typing import Dict
import dns.enum
import dns.exception
@classmethod
def _extra_from_text(cls, text):
    if text.find('-') >= 0:
        try:
            return cls[text.replace('-', '_')]
        except KeyError:
            pass
    return _registered_by_text.get(text)