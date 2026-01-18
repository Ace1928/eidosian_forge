import os
import warnings
import re
def _readmailcapfile(fp, lineno):
    """Read a mailcap file and return a dictionary keyed by MIME type.

    Each MIME type is mapped to an entry consisting of a list of
    dictionaries; the list will contain more than one such dictionary
    if a given MIME type appears more than once in the mailcap file.
    Each dictionary contains key-value pairs for that MIME type, where
    the viewing command is stored with the key "view".
    """
    caps = {}
    while 1:
        line = fp.readline()
        if not line:
            break
        if line[0] == '#' or line.strip() == '':
            continue
        nextline = line
        while nextline[-2:] == '\\\n':
            nextline = fp.readline()
            if not nextline:
                nextline = '\n'
            line = line[:-2] + nextline
        key, fields = parseline(line)
        if not (key and fields):
            continue
        if lineno is not None:
            fields['lineno'] = lineno
            lineno += 1
        types = key.split('/')
        for j in range(len(types)):
            types[j] = types[j].strip()
        key = '/'.join(types).lower()
        if key in caps:
            caps[key].append(fields)
        else:
            caps[key] = [fields]
    return (caps, lineno)