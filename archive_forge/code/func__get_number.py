import copy
import re
from collections import namedtuple
def _get_number(self, name, default_value=0):
    option_value = getattr(self.raw_options, name, default_value)
    result = 0
    try:
        result = int(option_value)
    except ValueError:
        pass
    return result