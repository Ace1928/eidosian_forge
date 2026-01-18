from __future__ import (absolute_import, division, print_function)
import fnmatch
import os
import sys
import re
import itertools
import traceback
from operator import attrgetter
from random import shuffle
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleOptionsError, AnsibleParserError
from ansible.inventory.data import InventoryData
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.parsing.utils.addresses import parse_address
from ansible.plugins.loader import inventory_loader
from ansible.utils.helpers import deduplicate_list
from ansible.utils.path import unfrackpath
from ansible.utils.display import Display
from ansible.utils.vars import combine_vars
from ansible.vars.plugins import get_vars_from_inventory_sources
def _match_one_pattern(self, pattern):
    """
        Takes a single pattern and returns a list of matching host names.
        Ignores intersection (&) and exclusion (!) specifiers.

        The pattern may be:

            1. A regex starting with ~, e.g. '~[abc]*'
            2. A shell glob pattern with ?/*/[chars]/[!chars], e.g. 'foo*'
            3. An ordinary word that matches itself only, e.g. 'foo'

        The pattern is matched using the following rules:

            1. If it's 'all', it matches all hosts in all groups.
            2. Otherwise, for each known group name:
                (a) if it matches the group name, the results include all hosts
                    in the group or any of its children.
                (b) otherwise, if it matches any hosts in the group, the results
                    include the matching hosts.

        This means that 'foo*' may match one or more groups (thus including all
        hosts therein) but also hosts in other groups.

        The built-in groups 'all' and 'ungrouped' are special. No pattern can
        match these group names (though 'all' behaves as though it matches, as
        described above). The word 'ungrouped' can match a host of that name,
        and patterns like 'ungr*' and 'al*' can match either hosts or groups
        other than all and ungrouped.

        If the pattern matches one or more group names according to these rules,
        it may have an optional range suffix to select a subset of the results.
        This is allowed only if the pattern is not a regex, i.e. '~foo[1]' does
        not work (the [1] is interpreted as part of the regex), but 'foo*[1]'
        would work if 'foo*' matched the name of one or more groups.

        Duplicate matches are always eliminated from the results.
        """
    if pattern[0] in ('&', '!'):
        pattern = pattern[1:]
    if pattern not in self._pattern_cache:
        expr, slice = self._split_subscript(pattern)
        hosts = self._enumerate_matches(expr)
        try:
            hosts = self._apply_subscript(hosts, slice)
        except IndexError:
            raise AnsibleError("No hosts matched the subscripted pattern '%s'" % pattern)
        self._pattern_cache[pattern] = hosts
    return self._pattern_cache[pattern]