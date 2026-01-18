import csv
import gzip
import json
from nltk.internals import deprecated
def _get_entity_recursive(json, entity):
    if not json:
        return None
    elif isinstance(json, dict):
        for key, value in json.items():
            if key == entity:
                return value
            if key == 'entities' or key == 'extended_entities':
                candidate = _get_entity_recursive(value, entity)
                if candidate is not None:
                    return candidate
        return None
    elif isinstance(json, list):
        for item in json:
            candidate = _get_entity_recursive(item, entity)
            if candidate is not None:
                return candidate
        return None
    else:
        return None