import re
from typing import Optional
def _strip_string(string):
    string = string.replace('\n', '')
    string = string.replace('\\!', '')
    string = string.replace('\\\\', '\\')
    string = string.replace('tfrac', 'frac')
    string = string.replace('dfrac', 'frac')
    string = string.replace('\\left', '')
    string = string.replace('\\right', '')
    string = string.replace('^{\\circ}', '')
    string = string.replace('^\\circ', '')
    string = string.replace('\\$', '')
    string = _remove_right_units(string)
    string = string.replace('\\%', '')
    string = string.replace('\\%', '')
    string = string.replace(' .', ' 0.')
    string = string.replace('{.', '{0.')
    if len(string) == 0:
        return string
    if string[0] == '.':
        string = '0' + string
    if len(string.split('=')) == 2:
        if len(string.split('=')[0]) <= 2:
            string = string.split('=')[1]
    string = _fix_sqrt(string)
    string = string.replace(' ', '')
    string = _fix_fracs(string)
    if string == '0.5':
        string = '\\frac{1}{2}'
    string = _fix_a_slash_b(string)
    return string