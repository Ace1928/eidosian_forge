import os
import re
import sys
from codecs import BOM_UTF8, BOM_UTF16, BOM_UTF16_BE, BOM_UTF16_LE
import six
from ._version import __version__
def _set_configspec(self, section, copy):
    """
        Called by validate. Handles setting the configspec on subsections
        including sections to be validated by __many__
        """
    configspec = section.configspec
    many = configspec.get('__many__')
    if isinstance(many, dict):
        for entry in section.sections:
            if entry not in configspec:
                section[entry].configspec = many
    for entry in configspec.sections:
        if entry == '__many__':
            continue
        if entry not in section:
            section[entry] = {}
            section[entry]._created = True
            if copy:
                section.comments[entry] = configspec.comments.get(entry, [])
                section.inline_comments[entry] = configspec.inline_comments.get(entry, '')
        if isinstance(section[entry], Section):
            section[entry].configspec = configspec[entry]