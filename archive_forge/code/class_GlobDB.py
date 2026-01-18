import os
import re
import stat
import sys
import fnmatch
from xdg import BaseDirectory
import xdg.Locale
from xml.dom import minidom, XML_NAMESPACE
from collections import defaultdict
class GlobDB(object):

    def __init__(self):
        """Prepare the GlobDB. It can't actually be used until .finalise() is
        called, but merge_file() can be used to add data before that.
        """
        self.allglobs = defaultdict(set)

    def merge_file(self, path):
        """Loads name matching information from a globs2 file."""
        allglobs = self.allglobs
        with open(path) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                fields = line[:-1].split(':')
                weight, type_name, pattern = fields[:3]
                weight = int(weight)
                mtype = lookup(type_name)
                if len(fields) > 3:
                    flags = fields[3].split(',')
                else:
                    flags = ()
                if pattern == '__NOGLOBS__':
                    allglobs.pop(mtype, None)
                    continue
                allglobs[mtype].add((weight, pattern, tuple(flags)))

    def finalise(self):
        """Prepare the GlobDB for matching.
        
        This should be called after all files have been merged into it.
        """
        self.exts = defaultdict(list)
        self.cased_exts = defaultdict(list)
        self.globs = []
        self.literals = {}
        self.cased_literals = {}
        for mtype, globs in self.allglobs.items():
            mtype = mtype.canonical()
            for weight, pattern, flags in globs:
                cased = 'cs' in flags
                if pattern.startswith('*.'):
                    rest = pattern[2:]
                    if not ('*' in rest or '[' in rest or '?' in rest):
                        if cased:
                            self.cased_exts[rest].append((mtype, weight))
                        else:
                            self.exts[rest.lower()].append((mtype, weight))
                        continue
                if '*' in pattern or '[' in pattern or '?' in pattern:
                    re_flags = 0 if cased else re.I
                    pattern = re.compile(fnmatch.translate(pattern), flags=re_flags)
                    self.globs.append((pattern, mtype, weight))
                elif cased:
                    self.cased_literals[pattern] = (mtype, weight)
                else:
                    self.literals[pattern.lower()] = (mtype, weight)
        self.globs.sort(reverse=True, key=lambda x: (x[2], len(x[0].pattern)))

    def first_match(self, path):
        """Return the first match found for a given path, or None if no match
        is found."""
        try:
            return next(self._match_path(path))[0]
        except StopIteration:
            return None

    def all_matches(self, path):
        """Return a list of (MIMEtype, glob weight) pairs for the path."""
        return list(self._match_path(path))

    def _match_path(self, path):
        """Yields pairs of (mimetype, glob weight)."""
        leaf = os.path.basename(path)
        if leaf in self.cased_literals:
            yield self.cased_literals[leaf]
        lleaf = leaf.lower()
        if lleaf in self.literals:
            yield self.literals[lleaf]
        ext = leaf
        while 1:
            p = ext.find('.')
            if p < 0:
                break
            ext = ext[p + 1:]
            if ext in self.cased_exts:
                for res in self.cased_exts[ext]:
                    yield res
        ext = lleaf
        while 1:
            p = ext.find('.')
            if p < 0:
                break
            ext = ext[p + 1:]
            if ext in self.exts:
                for res in self.exts[ext]:
                    yield res
        for regex, mime_type, weight in self.globs:
            if regex.match(leaf):
                yield (mime_type, weight)