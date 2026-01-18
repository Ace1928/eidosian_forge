import collections.abc
import datetime
import email.utils
import functools
import logging
import io
import re
import subprocess
import warnings
import chardet
from debian._util import (
from debian.deprecation import function_deprecated_by
import debian.debian_support
import debian.changelog
class TagSectionWrapper(_TagSectionWrapper_base):
    """Wrap a TagSection object, using its find_raw method to get field values

    This allows us to pick which whitespace to strip off the beginning and end
    of the data, so we don't lose leading newlines.
    """

    def __init__(self, section, decoder=None):
        self.__section = section
        self.decoder = decoder or _AutoDecoder()
        super(TagSectionWrapper, self).__init__()

    def __iter__(self):
        for key in self.__section.keys():
            if not key.startswith('#'):
                yield key

    def __len__(self):
        return len([key for key in self.__section.keys() if not key.startswith('#')])

    def __getitem__(self, key):
        sraw = self.__section.find_raw(key)
        s = self.decoder.decode(sraw)
        if s is None:
            raise KeyError(key)
        data = s[s.find(':') + 1:]
        return data.lstrip(' \t').rstrip('\n')