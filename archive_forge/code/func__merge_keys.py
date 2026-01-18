import logging
import re
from boto.vendored.regions.exceptions import NoRegionError
def _merge_keys(self, from_data, result):
    for key in from_data:
        if key not in result:
            result[key] = from_data[key]