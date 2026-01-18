import logging
import re
from boto.vendored.regions.exceptions import NoRegionError
def _region_match(self, partition, region_name):
    if region_name in partition['regions']:
        return True
    if 'regionRegex' in partition:
        return re.compile(partition['regionRegex']).match(region_name)
    return False