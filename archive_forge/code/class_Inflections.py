from __future__ import absolute_import, division, print_function
import errno
import glob
import json
import os
import re
import re
import re
class Inflections(object):
    """
    Inflections - rules how to convert words from singular to plural and vice versa.
    """

    def __init__(self):
        self.plurals = []
        self.singulars = []
        self.uncountables = []
        self.humans = []
        self.acronyms = {}
        self.acronym_regex = '/(?=a)b/'

    def acronym(self, word):
        """
        Add a new acronym.
        """
        self.acronyms[word.lower()] = word
        self.acronym_regex = '|'.join(self.acronyms.values())

    def plural(self, rule, replacement):
        """
        Add a new plural rule.
        """
        if rule in self.uncountables:
            self.uncountables.remove(rule)
        if replacement in self.uncountables:
            self.uncountables.remove(replacement)
        self.plurals.insert(0, (rule, replacement))

    def singular(self, rule, replacement):
        """
        Add a new singular rule.
        """
        if rule in self.uncountables:
            self.uncountables.remove(rule)
        if replacement in self.uncountables:
            self.uncountables.remove(replacement)
        self.singulars.insert(0, (rule, replacement))

    def irregular(self, singular, plural):
        """
        Add a new irregular rule
        """
        if singular in self.uncountables:
            self.uncountables.remove(singular)
        if plural in self.uncountables:
            self.uncountables.remove(plural)
        sfirst = singular[0]
        srest = singular[1:]
        pfirst = plural[0]
        prest = plural[1:]
        if sfirst.upper() == pfirst.upper():
            self.plural('(?i)({}){}$'.format(sfirst, srest), '\\1' + prest)
            self.plural('(?i)({}){}$'.format(pfirst, prest), '\\1' + prest)
            self.singular('(?i)({}){}$'.format(sfirst, srest), '\\1' + srest)
            self.singular('(?i)({}){}$'.format(pfirst, prest), '\\1' + srest)
        else:
            self.plural('{}(?i){}$'.format(sfirst.upper(), srest), pfirst.upper() + prest)
            self.plural('{}(?i){}$'.format(sfirst.lower(), srest), pfirst.lower() + prest)
            self.plural('{}(?i){}$'.format(pfirst.upper(), prest), pfirst.upper() + prest)
            self.plural('{}(?i){}$'.format(pfirst.lower(), prest), pfirst.lower() + prest)
            self.singular('{}(?i){}$'.format(sfirst.upper(), srest), sfirst.upper() + srest)
            self.singular('{}(?i){}$'.format(sfirst.lower(), srest), sfirst.lower() + srest)
            self.singular('{}(?i){}$'.format(pfirst.upper(), prest), sfirst.upper() + srest)
            self.singular('{}(?i){}$'.format(pfirst.lower(), prest), sfirst.lower() + srest)

    def uncountable(self, *words):
        """
        Add new uncountables.
        """
        self.uncountables.extend(words)

    def human(self, rule, replacement):
        """
        Add a new humanize rule.
        """
        self.humans.insert(0, (rule, replacement))