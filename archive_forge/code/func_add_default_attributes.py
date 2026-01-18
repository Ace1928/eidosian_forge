from __future__ import annotations
import re
from ansible.module_utils.compat.version import StrictVersion
from functools import partial
from urllib.parse import urlparse
from voluptuous import ALLOW_EXTRA, PREVENT_EXTRA, All, Any, Invalid, Length, MultipleInvalid, Required, Schema, Self, ValueInvalid, Exclusive
from ansible.constants import DOCUMENTABLE_PLUGINS
from ansible.module_utils.six import string_types
from ansible.module_utils.common.collections import is_iterable
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.parsing.quoting import unquote
from ansible.utils.version import SemanticVersion
from ansible.release import __version__
from antsibull_docs_parser import dom
from antsibull_docs_parser.parser import parse, Context
from .utils import parse_isodate
def add_default_attributes(more=None):
    schema = {'description': doc_string_or_strings, 'details': doc_string_or_strings, 'support': any_string_types, 'version_added_collection': any_string_types, 'version_added': any_string_types}
    if more:
        schema.update(more)
    return schema