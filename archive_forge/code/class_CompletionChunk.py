from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
class CompletionChunk(object):

    class Kind(object):

        def __init__(self, name):
            self.name = name

        def __str__(self):
            return self.name

        def __repr__(self):
            return '<ChunkKind: %s>' % self

    def __init__(self, completionString, key):
        self.cs = completionString
        self.key = key
        self.__kindNumberCache = -1

    def __repr__(self):
        return "{'" + self.spelling + "', " + str(self.kind) + '}'

    @CachedProperty
    def spelling(self):
        if self.__kindNumber in SpellingCache:
            return SpellingCache[self.__kindNumber]
        return conf.lib.clang_getCompletionChunkText(self.cs, self.key)

    @property
    def __kindNumber(self):
        if self.__kindNumberCache == -1:
            self.__kindNumberCache = conf.lib.clang_getCompletionChunkKind(self.cs, self.key)
        return self.__kindNumberCache

    @CachedProperty
    def kind(self):
        return completionChunkKindMap[self.__kindNumber]

    @CachedProperty
    def string(self):
        res = conf.lib.clang_getCompletionChunkCompletionString(self.cs, self.key)
        if res:
            return CompletionString(res)
        else:
            None

    def isKindOptional(self):
        return self.__kindNumber == 0

    def isKindTypedText(self):
        return self.__kindNumber == 1

    def isKindPlaceHolder(self):
        return self.__kindNumber == 3

    def isKindInformative(self):
        return self.__kindNumber == 4

    def isKindResultType(self):
        return self.__kindNumber == 15