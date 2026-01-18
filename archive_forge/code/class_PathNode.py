from __future__ import with_statement
import optparse
import sys
import tokenize
from collections import defaultdict
class PathNode(object):

    def __init__(self, name, look='circle'):
        self.name = name
        self.look = look

    def to_dot(self):
        print('node [shape=%s,label="%s"] %d;' % (self.look, self.name, self.dot_id()))

    def dot_id(self):
        return id(self)