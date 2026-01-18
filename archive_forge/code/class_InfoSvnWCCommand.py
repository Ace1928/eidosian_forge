import os, sys, time, re, calendar
import py
import subprocess
from py._path import common
class InfoSvnWCCommand:

    def __init__(self, output):
        d = {}
        for line in output.split('\n'):
            if not line.strip():
                continue
            key, value = line.split(':', 1)
            key = key.lower().replace(' ', '')
            value = value.strip()
            d[key] = value
        try:
            self.url = d['url']
        except KeyError:
            raise ValueError('Not a versioned resource')
        self.kind = d['nodekind'] == 'directory' and 'dir' or d['nodekind']
        try:
            self.rev = int(d['revision'])
        except KeyError:
            self.rev = None
        self.path = py.path.local(d['path'])
        self.size = self.path.size()
        if 'lastchangedrev' in d:
            self.created_rev = int(d['lastchangedrev'])
        if 'lastchangedauthor' in d:
            self.last_author = d['lastchangedauthor']
        if 'lastchangeddate' in d:
            self.mtime = parse_wcinfotime(d['lastchangeddate'])
            self.time = self.mtime * 1000000

    def __eq__(self, other):
        return self.__dict__ == other.__dict__