import functools
import re
import tokenize
from hacking import core
def _parse_assert_mock_str(line):
    point = line.find('.assert_')
    if point != -1:
        end_pos = line[point:].find('(') + point
        return (point, line[point + 1:end_pos], line[:point])
    else:
        return (None, None, None)