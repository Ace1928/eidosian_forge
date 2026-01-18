from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
class CompletionString(ClangObject):

    class Availability(object):

        def __init__(self, name):
            self.name = name

        def __str__(self):
            return self.name

        def __repr__(self):
            return '<Availability: %s>' % self

    def __len__(self):
        return self.num_chunks

    @CachedProperty
    def num_chunks(self):
        return conf.lib.clang_getNumCompletionChunks(self.obj)

    def __getitem__(self, key):
        if self.num_chunks <= key:
            raise IndexError
        return CompletionChunk(self.obj, key)

    @property
    def priority(self):
        return conf.lib.clang_getCompletionPriority(self.obj)

    @property
    def availability(self):
        res = conf.lib.clang_getCompletionAvailability(self.obj)
        return availabilityKinds[res]

    @property
    def briefComment(self):
        if conf.function_exists('clang_getCompletionBriefComment'):
            return conf.lib.clang_getCompletionBriefComment(self.obj)
        return _CXString()

    def __repr__(self):
        return ' | '.join([str(a) for a in self]) + ' || Priority: ' + str(self.priority) + ' || Availability: ' + str(self.availability) + ' || Brief comment: ' + str(self.briefComment)