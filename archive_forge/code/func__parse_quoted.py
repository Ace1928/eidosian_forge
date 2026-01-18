from __future__ import (absolute_import, division, print_function)
import sys
import warnings
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible_collections.community.dns.plugins.module_utils.conversion.base import (
def _parse_quoted(value, index, use_octal):
    if index == len(value):
        raise DNSConversionError(u'Unexpected backslash at end of string')
    letter = value[index:index + 1]
    index += 1
    if letter in (b'\\', b'"'):
        return (letter, index)
    v2 = _DECIMAL_DIGITS.find(letter)
    if v2 < 0 or (use_octal and v2 >= 8):
        raise DNSConversionError(u'A backslash must not be followed by "{letter}" (index {index})'.format(letter=to_text(letter), index=index))
    if index + 1 >= len(value):
        raise DNSConversionError(u'The {type} sequence at the end requires {missing} more digit(s)'.format(type='octal' if use_octal else 'decimal', missing=index + 2 - len(value)))
    letter = value[index:index + 1]
    index += 1
    v1 = _DECIMAL_DIGITS.find(letter)
    if v1 < 0 or (use_octal and v1 >= 8):
        raise DNSConversionError(u'The second letter of the {type} sequence at index {index} is not a {type} digit, but "{letter}"'.format(type='octal' if use_octal else 'decimal', letter=to_text(letter), index=index))
    letter = value[index:index + 1]
    index += 1
    v0 = _DECIMAL_DIGITS.find(letter)
    if v0 < 0 or (use_octal and v0 >= 8):
        raise DNSConversionError(u'The third letter of the {type} sequence at index {index} is not a {type} digit, but "{letter}"'.format(type='octal' if use_octal else 'decimal', letter=to_text(letter), index=index))
    if use_octal:
        return (_int_to_byte(v2 * 64 + v1 * 8 + v0), index)
    return (_int_to_byte(v2 * 100 + v1 * 10 + v0), index)