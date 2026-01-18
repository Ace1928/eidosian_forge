import os
import re
import stat
import sys
import fnmatch
from xdg import BaseDirectory
import xdg.Locale
from xml.dom import minidom, XML_NAMESPACE
from collections import defaultdict
class MagicMatchAny(object):
    """Match any of a set of magic rules.
    
    This has a similar interface to MagicRule objects (i.e. its match() and
    maxlen() methods), to allow for duck typing.
    """

    def __init__(self, rules):
        self.rules = rules

    def match(self, buffer):
        return any((r.match(buffer) for r in self.rules))

    def maxlen(self):
        return max((r.maxlen() for r in self.rules))

    @classmethod
    def from_file(cls, f):
        """Read a set of rules from the binary magic file."""
        c = f.read(1)
        f.seek(-1, 1)
        depths_rules = []
        while c and c != b'[':
            try:
                depths_rules.append(MagicRule.from_file(f))
            except UnknownMagicRuleFormat:
                pass
            c = f.read(1)
            if c:
                f.seek(-1, 1)
        tree = []
        insert_points = {0: tree}
        for depth, rule in depths_rules:
            subrules = []
            insert_points[depth].append((rule, subrules))
            insert_points[depth + 1] = subrules
        return cls.from_rule_tree(tree)

    @classmethod
    def from_rule_tree(cls, tree):
        """From a nested list of (rule, subrules) pairs, build a MagicMatchAny
        instance, recursing down the tree.
        
        Where there's only one top-level rule, this is returned directly,
        to simplify the nested structure. Returns None if no rules were read.
        """
        rules = []
        for rule, subrules in tree:
            if subrules:
                rule.also = cls.from_rule_tree(subrules)
            rules.append(rule)
        if len(rules) == 0:
            return None
        if len(rules) == 1:
            return rules[0]
        return cls(rules)