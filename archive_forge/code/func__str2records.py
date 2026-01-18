import os
import re
import shelve
import sys
import nltk.data
def _str2records(filename, rel):
    """
    Read a file into memory and convert each relation clause into a list.
    """
    recs = []
    contents = nltk.data.load('corpora/chat80/%s' % filename, format='text')
    for line in contents.splitlines():
        if line.startswith(rel):
            line = re.sub(rel + '\\(', '', line)
            line = re.sub('\\)\\.$', '', line)
            record = line.split(',')
            recs.append(record)
    return recs