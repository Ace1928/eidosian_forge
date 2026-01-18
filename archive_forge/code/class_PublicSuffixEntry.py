from __future__ import absolute_import, division, print_function
import os.path
import re
from ansible_collections.community.dns.plugins.module_utils.names import InvalidDomainName, split_into_labels, normalize_label
class PublicSuffixEntry(object):
    """
    Contains a Public Suffix List entry with metadata.
    """

    def __init__(self, labels, exception_rule=False, part=None):
        self.labels = labels
        self.exception_rule = exception_rule
        self.part = part

    def matches(self, normalized_labels):
        """
        Match PSL entry with a given normalized list of labels.
        """
        if len(normalized_labels) < len(self.labels):
            return False
        for i, label in enumerate(self.labels):
            normalized_label = normalized_labels[i]
            if label not in (normalized_label, '*'):
                return False
        return True