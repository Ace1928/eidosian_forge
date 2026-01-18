import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def UpdateProperties(self, properties, do_copy=False):
    """Merge the supplied properties into the _properties dictionary.

    The input properties must adhere to the class schema or a KeyError or
    TypeError exception will be raised.  If adding an object of an XCObject
    subclass and the schema indicates a strong relationship, the object's
    parent will be set to this object.

    If do_copy is True, then lists, dicts, strong-owned XCObjects, and
    strong-owned XCObjects in lists will be copied instead of having their
    references added.
    """
    if properties is None:
        return
    for property, value in properties.items():
        if property not in self._schema:
            raise KeyError(property + ' not in ' + self.__class__.__name__)
        is_list, property_type, is_strong = self._schema[property][0:3]
        if is_list:
            if value.__class__ != list:
                raise TypeError(property + ' of ' + self.__class__.__name__ + ' must be list, not ' + value.__class__.__name__)
            for item in value:
                if not isinstance(item, property_type) and (not (isinstance(item, str) and property_type == str)):
                    raise TypeError('item of ' + property + ' of ' + self.__class__.__name__ + ' must be ' + property_type.__name__ + ', not ' + item.__class__.__name__)
        elif not isinstance(value, property_type) and (not (isinstance(value, str) and property_type == str)):
            raise TypeError(property + ' of ' + self.__class__.__name__ + ' must be ' + property_type.__name__ + ', not ' + value.__class__.__name__)
        if do_copy:
            if isinstance(value, XCObject):
                if is_strong:
                    self._properties[property] = value.Copy()
                else:
                    self._properties[property] = value
            elif isinstance(value, (str, int)):
                self._properties[property] = value
            elif isinstance(value, list):
                if is_strong:
                    self._properties[property] = []
                    for item in value:
                        self._properties[property].append(item.Copy())
                else:
                    self._properties[property] = value[:]
            elif isinstance(value, dict):
                self._properties[property] = value.copy()
            else:
                raise TypeError("Don't know how to copy a " + value.__class__.__name__ + ' object for ' + property + ' in ' + self.__class__.__name__)
        else:
            self._properties[property] = value
        if is_strong:
            if not is_list:
                self._properties[property].parent = self
            else:
                for item in self._properties[property]:
                    item.parent = self