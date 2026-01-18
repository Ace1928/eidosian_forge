import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def _parseNoCache(self, instring, loc, doActions=True, callPreParse=True):
    debugging = self.debug
    if debugging or self.failAction:
        if self.debugActions[0]:
            self.debugActions[0](instring, loc, self)
        if callPreParse and self.callPreparse:
            preloc = self.preParse(instring, loc)
        else:
            preloc = loc
        tokensStart = preloc
        try:
            try:
                loc, tokens = self.parseImpl(instring, preloc, doActions)
            except IndexError:
                raise ParseException(instring, len(instring), self.errmsg, self)
        except ParseBaseException as err:
            if self.debugActions[2]:
                self.debugActions[2](instring, tokensStart, self, err)
            if self.failAction:
                self.failAction(instring, tokensStart, self, err)
            raise
    else:
        if callPreParse and self.callPreparse:
            preloc = self.preParse(instring, loc)
        else:
            preloc = loc
        tokensStart = preloc
        if self.mayIndexError or loc >= len(instring):
            try:
                loc, tokens = self.parseImpl(instring, preloc, doActions)
            except IndexError:
                raise ParseException(instring, len(instring), self.errmsg, self)
        else:
            loc, tokens = self.parseImpl(instring, preloc, doActions)
    tokens = self.postParse(instring, loc, tokens)
    retTokens = ParseResults(tokens, self.resultsName, asList=self.saveAsList, modal=self.modalResults)
    if self.parseAction and (doActions or self.callDuringTry):
        if debugging:
            try:
                for fn in self.parseAction:
                    tokens = fn(instring, tokensStart, retTokens)
                    if tokens is not None:
                        retTokens = ParseResults(tokens, self.resultsName, asList=self.saveAsList and isinstance(tokens, (ParseResults, list)), modal=self.modalResults)
            except ParseBaseException as err:
                if self.debugActions[2]:
                    self.debugActions[2](instring, tokensStart, self, err)
                raise
        else:
            for fn in self.parseAction:
                tokens = fn(instring, tokensStart, retTokens)
                if tokens is not None:
                    retTokens = ParseResults(tokens, self.resultsName, asList=self.saveAsList and isinstance(tokens, (ParseResults, list)), modal=self.modalResults)
    if debugging:
        if self.debugActions[1]:
            self.debugActions[1](instring, tokensStart, loc, self, retTokens)
    return (loc, retTokens)