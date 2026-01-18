import copy
import json
import jsonpatch
import warlock.model as warlock
def _check_property(self, property_name, allow_non_base):
    for prop in self.properties:
        if property_name == prop.name:
            return prop.is_base or allow_non_base
    return False