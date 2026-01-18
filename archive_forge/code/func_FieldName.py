from __future__ import print_function
from __future__ import unicode_literals
import collections
import contextlib
import gzip
import json
import keyword
import logging
import os
import re
import tempfile
import six
from six.moves import urllib_parse
import six.moves.urllib.error as urllib_error
import six.moves.urllib.request as urllib_request
def FieldName(self, name):
    """Generate a valid field name from name."""
    name = self.__StripName(name)
    if self.__name_convention == 'LOWER_CAMEL':
        name = Names.__ToLowerCamel(name)
    elif self.__name_convention == 'LOWER_WITH_UNDER':
        name = Names.__FromCamel(name)
    return Names.CleanName(name)