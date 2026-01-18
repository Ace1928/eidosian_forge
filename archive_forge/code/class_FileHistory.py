from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
import datetime
import os
class FileHistory(History):
    """
    :class:`.History` class that stores all strings in a file.
    """

    def __init__(self, filename):
        self.strings = []
        self.filename = filename
        self._load()

    def _load(self):
        lines = []

        def add():
            if lines:
                string = ''.join(lines)[:-1]
                self.strings.append(string)
        if os.path.exists(self.filename):
            with open(self.filename, 'rb') as f:
                for line in f:
                    line = line.decode('utf-8')
                    if line.startswith('+'):
                        lines.append(line[1:])
                    else:
                        add()
                        lines = []
                add()

    def append(self, string):
        self.strings.append(string)
        with open(self.filename, 'ab') as f:

            def write(t):
                f.write(t.encode('utf-8'))
            write('\n# %s\n' % datetime.datetime.now())
            for line in string.split('\n'):
                write('+%s\n' % line)

    def __getitem__(self, key):
        return self.strings[key]

    def __iter__(self):
        return iter(self.strings)

    def __len__(self):
        return len(self.strings)