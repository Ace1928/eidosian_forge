import os as _os
import sys as _sys
import warnings as _warnings
from .base import Sign
from .controller_db import mapping_list
def _parse_mapping(mapping_string):
    """Parse a SDL2 style GameController mapping string.

    :Parameters:
        `mapping_string` : str
            A raw string containing an SDL style controller mapping.

    :rtype: A dict containing axis/button mapping relations.
    """
    valid_keys = ['guide', 'back', 'start', 'a', 'b', 'x', 'y', 'leftshoulder', 'leftstick', 'rightshoulder', 'rightstick', 'dpup', 'dpdown', 'dpleft', 'dpright', 'lefttrigger', 'righttrigger', 'leftx', 'lefty', 'rightx', 'righty']
    split_mapping = mapping_string.strip().split(',')
    relations = dict(guid=split_mapping[0], name=split_mapping[1])
    for item in split_mapping[2:]:
        if ':' not in item:
            continue
        key, relation_string, *etc = item.split(':')
        if key not in valid_keys:
            continue
        if '+' in relation_string:
            relation_string = relation_string.strip('+')
            sign = Sign.POSITIVE
        elif '-' in relation_string:
            relation_string = relation_string.strip('-')
            sign = Sign.NEGATIVE
        elif '~' in relation_string:
            relation_string = relation_string.strip('~')
            sign = Sign.INVERTED
        else:
            sign = Sign.DEFAULT
        if relation_string.startswith('b'):
            relations[key] = Relation('button', int(relation_string[1:]), sign)
        elif relation_string.startswith('a'):
            relations[key] = Relation('axis', int(relation_string[1:]), sign)
        elif relation_string.startswith('h0'):
            relations[key] = Relation('hat0', int(relation_string.split('.')[1]), sign)
        else:
            _warnings.warn(f"Skipping unknown relation type: '{relation_string}'")
    return relations