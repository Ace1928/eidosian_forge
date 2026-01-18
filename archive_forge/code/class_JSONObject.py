import json
import os
import sys
class JSONObject(object):
    """Wrapper for a JSON object.

    Presents a JSON object as a python object (where fields are attributes)
    instead of a dictionary.  Undefined attributes produce a value of None
    instead of an AttributeError.

    Note that attribute names beginning with an underscore are excluded.
    """

    def __getattr__(self, attr):
        return None

    def to_dict(self):
        result = {}
        for attr, val in self.__dict__.iteritems():
            if not attr.startswith('_'):
                result[attr] = _make_serializable(val)
        return result
    ToDict = to_dict