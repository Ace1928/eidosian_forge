import collections
import itertools
import logging
import io
import re
from debian._deb822_repro import (
from debian.deb822 import RestrictedField, RestrictedFieldError
def add_files_paragraph(self, paragraph):
    """Adds a FilesParagraph to this object.

        The paragraph is inserted directly after the last FilesParagraph (which
        might be before a standalone LicenseParagraph).
        """
    if not isinstance(paragraph, FilesParagraph):
        raise TypeError('paragraph must be a FilesParagraph instance')
    last_i = -1
    for i, p in enumerate(self.__paragraphs):
        if isinstance(p, FilesParagraph):
            last_i = i
    self.__paragraphs.insert(last_i + 1, paragraph)
    self.__file.insert(last_i + 2, paragraph._underlying_paragraph)