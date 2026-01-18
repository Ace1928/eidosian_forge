from __future__ import absolute_import
import os
def ParseOptions(self):
    """Parses the 'options' field and sets appropriate fields."""
    if self.options:
        options = [option.strip() for option in self.options.split(',')]
    else:
        options = []
    for option in options:
        if option not in VALID_OPTIONS:
            raise BadConfig('Unrecognized option: %s', option)
    self.public = PUBLIC in options
    self.dynamic = DYNAMIC in options
    self.failfast = FAILFAST in options
    return self