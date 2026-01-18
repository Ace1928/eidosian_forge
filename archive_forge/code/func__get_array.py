import copy
import re
from collections import namedtuple
def _get_array(self, name, default_value=[]):
    option_value = getattr(self.raw_options, name, default_value)
    result = []
    if isinstance(option_value, list):
        result = copy.copy(option_value)
    elif isinstance(option_value, str):
        result = re.compile('[^a-zA-Z0-9_/\\-]+').split(option_value)
    return result