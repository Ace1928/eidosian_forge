import os
import re
import stat
import sys
import fnmatch
from xdg import BaseDirectory
import xdg.Locale
from xml.dom import minidom, XML_NAMESPACE
from collections import defaultdict
class MagicDB:

    def __init__(self):
        self.bytype = defaultdict(list)

    def merge_file(self, fname):
        """Read a magic binary file, and add its rules to this MagicDB."""
        with open(fname, 'rb') as f:
            line = f.readline()
            if line != b'MIME-Magic\x00\n':
                raise IOError('Not a MIME magic file')
            while True:
                shead = f.readline().decode('ascii')
                if not shead:
                    break
                if shead[0] != '[' or shead[-2:] != ']\n':
                    raise ValueError('Malformed section heading', shead)
                pri, tname = shead[1:-2].split(':')
                pri = int(pri)
                mtype = lookup(tname)
                try:
                    rule = MagicMatchAny.from_file(f)
                except DiscardMagicRules:
                    self.bytype.pop(mtype, None)
                    rule = MagicMatchAny.from_file(f)
                if rule is None:
                    continue
                self.bytype[mtype].append((pri, rule))

    def finalise(self):
        """Prepare the MagicDB for matching.
        
        This should be called after all rules have been merged into it.
        """
        maxlen = 0
        self.alltypes = []
        for mtype, rules in self.bytype.items():
            for pri, rule in rules:
                self.alltypes.append((pri, mtype, rule))
                maxlen = max(maxlen, rule.maxlen())
        self.maxlen = maxlen
        self.alltypes.sort(key=lambda x: x[0], reverse=True)

    def match_data(self, data, max_pri=100, min_pri=0, possible=None):
        """Do magic sniffing on some bytes.
        
        max_pri & min_pri can be used to specify the maximum & minimum priority
        rules to look for. possible can be a list of mimetypes to check, or None
        (the default) to check all mimetypes until one matches.
        
        Returns the MIMEtype found, or None if no entries match.
        """
        if possible is not None:
            types = []
            for mt in possible:
                for pri, rule in self.bytype[mt]:
                    types.append((pri, mt, rule))
            types.sort(key=lambda x: x[0])
        else:
            types = self.alltypes
        for priority, mimetype, rule in types:
            if priority > max_pri:
                continue
            if priority < min_pri:
                break
            if rule.match(data):
                return mimetype

    def match(self, path, max_pri=100, min_pri=0, possible=None):
        """Read data from the file and do magic sniffing on it.
        
        max_pri & min_pri can be used to specify the maximum & minimum priority
        rules to look for. possible can be a list of mimetypes to check, or None
        (the default) to check all mimetypes until one matches.
        
        Returns the MIMEtype found, or None if no entries match. Raises IOError
        if the file can't be opened.
        """
        with open(path, 'rb') as f:
            buf = f.read(self.maxlen)
        return self.match_data(buf, max_pri, min_pri, possible)

    def __repr__(self):
        return '<MagicDB (%d types)>' % len(self.alltypes)