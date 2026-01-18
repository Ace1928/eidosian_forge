from __future__ import (absolute_import, division, print_function)
import json
import re
from datetime import timedelta
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.six.moves.urllib.parse import urlparse
class DifferenceTracker(object):

    def __init__(self):
        self._diff = []

    def add(self, name, parameter=None, active=None):
        self._diff.append(dict(name=name, parameter=parameter, active=active))

    def merge(self, other_tracker):
        self._diff.extend(other_tracker._diff)

    @property
    def empty(self):
        return len(self._diff) == 0

    def get_before_after(self):
        """
        Return texts ``before`` and ``after``.
        """
        before = dict()
        after = dict()
        for item in self._diff:
            before[item['name']] = item['active']
            after[item['name']] = item['parameter']
        return (before, after)

    def has_difference_for(self, name):
        """
        Returns a boolean if a difference exists for name
        """
        return any((diff for diff in self._diff if diff['name'] == name))

    def get_legacy_docker_container_diffs(self):
        """
        Return differences in the docker_container legacy format.
        """
        result = []
        for entry in self._diff:
            item = dict()
            item[entry['name']] = dict(parameter=entry['parameter'], container=entry['active'])
            result.append(item)
        return result

    def get_legacy_docker_diffs(self):
        """
        Return differences in the docker_container legacy format.
        """
        result = [entry['name'] for entry in self._diff]
        return result