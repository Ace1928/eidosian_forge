import sys
import time
import prettytable
from cinderclient import exceptions
from cinderclient import utils
def _build_internal_dict(content):
    result = {}
    for pair in content.split(','):
        k, v = pair.split(':', 1)
        result.update({k.strip(): v.strip()})
    return result