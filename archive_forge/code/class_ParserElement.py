import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class ParserElement(object):
    """Abstract base level parser element class."""
    DEFAULT_WHITE_CHARS = ' \n\t\r'
    verbose_stacktrace = False

    def setDefaultWhitespaceChars(chars):
        """Overrides the default whitespace chars
        """
        ParserElement.DEFAULT_WHITE_CHARS = chars
    setDefaultWhitespaceChars = staticmethod(setDefaultWhitespaceChars)

    def inlineLiteralsUsing(cls):
        """
        Set class to be used for inclusion of string literals into a parser.
        """
        ParserElement.literalStringClass = cls
    inlineLiteralsUsing = staticmethod(inlineLiteralsUsing)

    def __init__(self, savelist=False):
        self.parseAction = list()
        self.failAction = None
        self.strRepr = None
        self.resultsName = None
        self.saveAsList = savelist
        self.skipWhitespace = True
        self.whiteChars = ParserElement.DEFAULT_WHITE_CHARS
        self.copyDefaultWhiteChars = True
        self.mayReturnEmpty = False
        self.keepTabs = False
        self.ignoreExprs = list()
        self.debug = False
        self.streamlined = False
        self.mayIndexError = True
        self.errmsg = ''
        self.modalResults = True
        self.debugActions = (None, None, None)
        self.re = None
        self.callPreparse = True
        self.callDuringTry = False

    def copy(self):
        """Make a copy of this C{ParserElement}.  Useful for defining different parse actions
           for the same parsing pattern, using copies of the original parse element."""
        cpy = copy.copy(self)
        cpy.parseAction = self.parseAction[:]
        cpy.ignoreExprs = self.ignoreExprs[:]
        if self.copyDefaultWhiteChars:
            cpy.whiteChars = ParserElement.DEFAULT_WHITE_CHARS
        return cpy

    def setName(self, name):
        """Define name for this expression, for use in debugging."""
        self.name = name
        self.errmsg = 'Expected ' + self.name
        if hasattr(self, 'exception'):
            self.exception.msg = self.errmsg
        return self

    def setResultsName(self, name, listAllMatches=False):
        """Define name for referencing matching tokens as a nested attribute
           of the returned parse results.
           NOTE: this returns a *copy* of the original C{ParserElement} object;
           this is so that the client can define a basic element, such as an
           integer, and reference it in multiple places with different names.
           
           You can also set results names using the abbreviated syntax,
           C{expr("name")} in place of C{expr.setResultsName("name")} - 
           see L{I{__call__}<__call__>}.
        """
        newself = self.copy()
        if name.endswith('*'):
            name = name[:-1]
            listAllMatches = True
        newself.resultsName = name
        newself.modalResults = not listAllMatches
        return newself

    def setBreak(self, breakFlag=True):
        """Method to invoke the Python pdb debugger when this element is
           about to be parsed. Set C{breakFlag} to True to enable, False to
           disable.
        """
        if breakFlag:
            _parseMethod = self._parse

            def breaker(instring, loc, doActions=True, callPreParse=True):
                import pdb
                pdb.set_trace()
                return _parseMethod(instring, loc, doActions, callPreParse)
            breaker._originalParseMethod = _parseMethod
            self._parse = breaker
        elif hasattr(self._parse, '_originalParseMethod'):
            self._parse = self._parse._originalParseMethod
        return self

    def setParseAction(self, *fns, **kwargs):
        """Define action to perform when successfully matching parse element definition.
           Parse action fn is a callable method with 0-3 arguments, called as C{fn(s,loc,toks)},
           C{fn(loc,toks)}, C{fn(toks)}, or just C{fn()}, where:
            - s   = the original string being parsed (see note below)
            - loc = the location of the matching substring
            - toks = a list of the matched tokens, packaged as a C{L{ParseResults}} object
           If the functions in fns modify the tokens, they can return them as the return
           value from fn, and the modified list of tokens will replace the original.
           Otherwise, fn does not need to return any value.

           Note: the default parsing behavior is to expand tabs in the input string
           before starting the parsing process.  See L{I{parseString}<parseString>} for more information
           on parsing strings containing C{<TAB>}s, and suggested methods to maintain a
           consistent view of the parsed string, the parse location, and line and column
           positions within the parsed string.
           """
        self.parseAction = list(map(_trim_arity, list(fns)))
        self.callDuringTry = 'callDuringTry' in kwargs and kwargs['callDuringTry']
        return self

    def addParseAction(self, *fns, **kwargs):
        """Add parse action to expression's list of parse actions. See L{I{setParseAction}<setParseAction>}."""
        self.parseAction += list(map(_trim_arity, list(fns)))
        self.callDuringTry = self.callDuringTry or ('callDuringTry' in kwargs and kwargs['callDuringTry'])
        return self

    def setFailAction(self, fn):
        """Define action to perform if parsing fails at this expression.
           Fail acton fn is a callable function that takes the arguments
           C{fn(s,loc,expr,err)} where:
            - s = string being parsed
            - loc = location where expression match was attempted and failed
            - expr = the parse expression that failed
            - err = the exception thrown
           The function returns no value.  It may throw C{L{ParseFatalException}}
           if it is desired to stop parsing immediately."""
        self.failAction = fn
        return self

    def _skipIgnorables(self, instring, loc):
        exprsFound = True
        while exprsFound:
            exprsFound = False
            for e in self.ignoreExprs:
                try:
                    while 1:
                        loc, dummy = e._parse(instring, loc)
                        exprsFound = True
                except ParseException:
                    pass
        return loc

    def preParse(self, instring, loc):
        if self.ignoreExprs:
            loc = self._skipIgnorables(instring, loc)
        if self.skipWhitespace:
            wt = self.whiteChars
            instrlen = len(instring)
            while loc < instrlen and instring[loc] in wt:
                loc += 1
        return loc

    def parseImpl(self, instring, loc, doActions=True):
        return (loc, [])

    def postParse(self, instring, loc, tokenlist):
        return tokenlist

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

    def tryParse(self, instring, loc):
        try:
            return self._parse(instring, loc, doActions=False)[0]
        except ParseFatalException:
            raise ParseException(instring, loc, self.errmsg, self)

    def _parseCache(self, instring, loc, doActions=True, callPreParse=True):
        lookup = (self, instring, loc, callPreParse, doActions)
        if lookup in ParserElement._exprArgCache:
            value = ParserElement._exprArgCache[lookup]
            if isinstance(value, Exception):
                raise value
            return (value[0], value[1].copy())
        else:
            try:
                value = self._parseNoCache(instring, loc, doActions, callPreParse)
                ParserElement._exprArgCache[lookup] = (value[0], value[1].copy())
                return value
            except ParseBaseException as pe:
                pe.__traceback__ = None
                ParserElement._exprArgCache[lookup] = pe
                raise
    _parse = _parseNoCache
    _exprArgCache = {}

    def resetCache():
        ParserElement._exprArgCache.clear()
    resetCache = staticmethod(resetCache)
    _packratEnabled = False

    def enablePackrat():
        """Enables "packrat" parsing, which adds memoizing to the parsing logic.
           Repeated parse attempts at the same string location (which happens
           often in many complex grammars) can immediately return a cached value,
           instead of re-executing parsing/validating code.  Memoizing is done of
           both valid results and parsing exceptions.

           This speedup may break existing programs that use parse actions that
           have side-effects.  For this reason, packrat parsing is disabled when
           you first import pyparsing.  To activate the packrat feature, your
           program must call the class method C{ParserElement.enablePackrat()}.  If
           your program uses C{psyco} to "compile as you go", you must call
           C{enablePackrat} before calling C{psyco.full()}.  If you do not do this,
           Python will crash.  For best results, call C{enablePackrat()} immediately
           after importing pyparsing.
        """
        if not ParserElement._packratEnabled:
            ParserElement._packratEnabled = True
            ParserElement._parse = ParserElement._parseCache
    enablePackrat = staticmethod(enablePackrat)

    def parseString(self, instring, parseAll=False):
        """Execute the parse expression with the given string.
           This is the main interface to the client code, once the complete
           expression has been built.

           If you want the grammar to require that the entire input string be
           successfully parsed, then set C{parseAll} to True (equivalent to ending
           the grammar with C{L{StringEnd()}}).

           Note: C{parseString} implicitly calls C{expandtabs()} on the input string,
           in order to report proper column numbers in parse actions.
           If the input string contains tabs and
           the grammar uses parse actions that use the C{loc} argument to index into the
           string being parsed, you can ensure you have a consistent view of the input
           string by:
            - calling C{parseWithTabs} on your grammar before calling C{parseString}
              (see L{I{parseWithTabs}<parseWithTabs>})
            - define your parse action using the full C{(s,loc,toks)} signature, and
              reference the input string using the parse action's C{s} argument
            - explictly expand the tabs in your input string before calling
              C{parseString}
        """
        ParserElement.resetCache()
        if not self.streamlined:
            self.streamline()
        for e in self.ignoreExprs:
            e.streamline()
        if not self.keepTabs:
            instring = instring.expandtabs()
        try:
            loc, tokens = self._parse(instring, 0)
            if parseAll:
                loc = self.preParse(instring, loc)
                se = Empty() + StringEnd()
                se._parse(instring, loc)
        except ParseBaseException as exc:
            if ParserElement.verbose_stacktrace:
                raise
            else:
                raise exc
        else:
            return tokens

    def scanString(self, instring, maxMatches=_MAX_INT, overlap=False):
        """Scan the input string for expression matches.  Each match will return the
           matching tokens, start location, and end location.  May be called with optional
           C{maxMatches} argument, to clip scanning after 'n' matches are found.  If
           C{overlap} is specified, then overlapping matches will be reported.

           Note that the start and end locations are reported relative to the string
           being parsed.  See L{I{parseString}<parseString>} for more information on parsing
           strings with embedded tabs."""
        if not self.streamlined:
            self.streamline()
        for e in self.ignoreExprs:
            e.streamline()
        if not self.keepTabs:
            instring = _ustr(instring).expandtabs()
        instrlen = len(instring)
        loc = 0
        preparseFn = self.preParse
        parseFn = self._parse
        ParserElement.resetCache()
        matches = 0
        try:
            while loc <= instrlen and matches < maxMatches:
                try:
                    preloc = preparseFn(instring, loc)
                    nextLoc, tokens = parseFn(instring, preloc, callPreParse=False)
                except ParseException:
                    loc = preloc + 1
                else:
                    if nextLoc > loc:
                        matches += 1
                        yield (tokens, preloc, nextLoc)
                        if overlap:
                            nextloc = preparseFn(instring, loc)
                            if nextloc > loc:
                                loc = nextLoc
                            else:
                                loc += 1
                        else:
                            loc = nextLoc
                    else:
                        loc = preloc + 1
        except ParseBaseException as exc:
            if ParserElement.verbose_stacktrace:
                raise
            else:
                raise exc

    def transformString(self, instring):
        """Extension to C{L{scanString}}, to modify matching text with modified tokens that may
           be returned from a parse action.  To use C{transformString}, define a grammar and
           attach a parse action to it that modifies the returned token list.
           Invoking C{transformString()} on a target string will then scan for matches,
           and replace the matched text patterns according to the logic in the parse
           action.  C{transformString()} returns the resulting transformed string."""
        out = []
        lastE = 0
        self.keepTabs = True
        try:
            for t, s, e in self.scanString(instring):
                out.append(instring[lastE:s])
                if t:
                    if isinstance(t, ParseResults):
                        out += t.asList()
                    elif isinstance(t, list):
                        out += t
                    else:
                        out.append(t)
                lastE = e
            out.append(instring[lastE:])
            out = [o for o in out if o]
            return ''.join(map(_ustr, _flatten(out)))
        except ParseBaseException as exc:
            if ParserElement.verbose_stacktrace:
                raise
            else:
                raise exc

    def searchString(self, instring, maxMatches=_MAX_INT):
        """Another extension to C{L{scanString}}, simplifying the access to the tokens found
           to match the given parse expression.  May be called with optional
           C{maxMatches} argument, to clip searching after 'n' matches are found.
        """
        try:
            return ParseResults([t for t, s, e in self.scanString(instring, maxMatches)])
        except ParseBaseException as exc:
            if ParserElement.verbose_stacktrace:
                raise
            else:
                raise exc

    def __add__(self, other):
        """Implementation of + operator - returns C{L{And}}"""
        if isinstance(other, basestring):
            other = ParserElement.literalStringClass(other)
        if not isinstance(other, ParserElement):
            warnings.warn('Cannot combine element of type %s with ParserElement' % type(other), SyntaxWarning, stacklevel=2)
            return None
        return And([self, other])

    def __radd__(self, other):
        """Implementation of + operator when left operand is not a C{L{ParserElement}}"""
        if isinstance(other, basestring):
            other = ParserElement.literalStringClass(other)
        if not isinstance(other, ParserElement):
            warnings.warn('Cannot combine element of type %s with ParserElement' % type(other), SyntaxWarning, stacklevel=2)
            return None
        return other + self

    def __sub__(self, other):
        """Implementation of - operator, returns C{L{And}} with error stop"""
        if isinstance(other, basestring):
            other = ParserElement.literalStringClass(other)
        if not isinstance(other, ParserElement):
            warnings.warn('Cannot combine element of type %s with ParserElement' % type(other), SyntaxWarning, stacklevel=2)
            return None
        return And([self, And._ErrorStop(), other])

    def __rsub__(self, other):
        """Implementation of - operator when left operand is not a C{L{ParserElement}}"""
        if isinstance(other, basestring):
            other = ParserElement.literalStringClass(other)
        if not isinstance(other, ParserElement):
            warnings.warn('Cannot combine element of type %s with ParserElement' % type(other), SyntaxWarning, stacklevel=2)
            return None
        return other - self

    def __mul__(self, other):
        """Implementation of * operator, allows use of C{expr * 3} in place of
           C{expr + expr + expr}.  Expressions may also me multiplied by a 2-integer
           tuple, similar to C{{min,max}} multipliers in regular expressions.  Tuples
           may also include C{None} as in:
            - C{expr*(n,None)} or C{expr*(n,)} is equivalent
              to C{expr*n + L{ZeroOrMore}(expr)}
              (read as "at least n instances of C{expr}")
            - C{expr*(None,n)} is equivalent to C{expr*(0,n)}
              (read as "0 to n instances of C{expr}")
            - C{expr*(None,None)} is equivalent to C{L{ZeroOrMore}(expr)}
            - C{expr*(1,None)} is equivalent to C{L{OneOrMore}(expr)}

           Note that C{expr*(None,n)} does not raise an exception if
           more than n exprs exist in the input stream; that is,
           C{expr*(None,n)} does not enforce a maximum number of expr
           occurrences.  If this behavior is desired, then write
           C{expr*(None,n) + ~expr}

        """
        if isinstance(other, int):
            minElements, optElements = (other, 0)
        elif isinstance(other, tuple):
            other = (other + (None, None))[:2]
            if other[0] is None:
                other = (0, other[1])
            if isinstance(other[0], int) and other[1] is None:
                if other[0] == 0:
                    return ZeroOrMore(self)
                if other[0] == 1:
                    return OneOrMore(self)
                else:
                    return self * other[0] + ZeroOrMore(self)
            elif isinstance(other[0], int) and isinstance(other[1], int):
                minElements, optElements = other
                optElements -= minElements
            else:
                raise TypeError("cannot multiply 'ParserElement' and ('%s','%s') objects", type(other[0]), type(other[1]))
        else:
            raise TypeError("cannot multiply 'ParserElement' and '%s' objects", type(other))
        if minElements < 0:
            raise ValueError('cannot multiply ParserElement by negative value')
        if optElements < 0:
            raise ValueError('second tuple value must be greater or equal to first tuple value')
        if minElements == optElements == 0:
            raise ValueError('cannot multiply ParserElement by 0 or (0,0)')
        if optElements:

            def makeOptionalList(n):
                if n > 1:
                    return Optional(self + makeOptionalList(n - 1))
                else:
                    return Optional(self)
            if minElements:
                if minElements == 1:
                    ret = self + makeOptionalList(optElements)
                else:
                    ret = And([self] * minElements) + makeOptionalList(optElements)
            else:
                ret = makeOptionalList(optElements)
        elif minElements == 1:
            ret = self
        else:
            ret = And([self] * minElements)
        return ret

    def __rmul__(self, other):
        return self.__mul__(other)

    def __or__(self, other):
        """Implementation of | operator - returns C{L{MatchFirst}}"""
        if isinstance(other, basestring):
            other = ParserElement.literalStringClass(other)
        if not isinstance(other, ParserElement):
            warnings.warn('Cannot combine element of type %s with ParserElement' % type(other), SyntaxWarning, stacklevel=2)
            return None
        return MatchFirst([self, other])

    def __ror__(self, other):
        """Implementation of | operator when left operand is not a C{L{ParserElement}}"""
        if isinstance(other, basestring):
            other = ParserElement.literalStringClass(other)
        if not isinstance(other, ParserElement):
            warnings.warn('Cannot combine element of type %s with ParserElement' % type(other), SyntaxWarning, stacklevel=2)
            return None
        return other | self

    def __xor__(self, other):
        """Implementation of ^ operator - returns C{L{Or}}"""
        if isinstance(other, basestring):
            other = ParserElement.literalStringClass(other)
        if not isinstance(other, ParserElement):
            warnings.warn('Cannot combine element of type %s with ParserElement' % type(other), SyntaxWarning, stacklevel=2)
            return None
        return Or([self, other])

    def __rxor__(self, other):
        """Implementation of ^ operator when left operand is not a C{L{ParserElement}}"""
        if isinstance(other, basestring):
            other = ParserElement.literalStringClass(other)
        if not isinstance(other, ParserElement):
            warnings.warn('Cannot combine element of type %s with ParserElement' % type(other), SyntaxWarning, stacklevel=2)
            return None
        return other ^ self

    def __and__(self, other):
        """Implementation of & operator - returns C{L{Each}}"""
        if isinstance(other, basestring):
            other = ParserElement.literalStringClass(other)
        if not isinstance(other, ParserElement):
            warnings.warn('Cannot combine element of type %s with ParserElement' % type(other), SyntaxWarning, stacklevel=2)
            return None
        return Each([self, other])

    def __rand__(self, other):
        """Implementation of & operator when left operand is not a C{L{ParserElement}}"""
        if isinstance(other, basestring):
            other = ParserElement.literalStringClass(other)
        if not isinstance(other, ParserElement):
            warnings.warn('Cannot combine element of type %s with ParserElement' % type(other), SyntaxWarning, stacklevel=2)
            return None
        return other & self

    def __invert__(self):
        """Implementation of ~ operator - returns C{L{NotAny}}"""
        return NotAny(self)

    def __call__(self, name):
        """Shortcut for C{L{setResultsName}}, with C{listAllMatches=default}::
             userdata = Word(alphas).setResultsName("name") + Word(nums+"-").setResultsName("socsecno")
           could be written as::
             userdata = Word(alphas)("name") + Word(nums+"-")("socsecno")
             
           If C{name} is given with a trailing C{'*'} character, then C{listAllMatches} will be
           passed as C{True}.
           """
        return self.setResultsName(name)

    def suppress(self):
        """Suppresses the output of this C{ParserElement}; useful to keep punctuation from
           cluttering up returned output.
        """
        return Suppress(self)

    def leaveWhitespace(self):
        """Disables the skipping of whitespace before matching the characters in the
           C{ParserElement}'s defined pattern.  This is normally only used internally by
           the pyparsing module, but may be needed in some whitespace-sensitive grammars.
        """
        self.skipWhitespace = False
        return self

    def setWhitespaceChars(self, chars):
        """Overrides the default whitespace chars
        """
        self.skipWhitespace = True
        self.whiteChars = chars
        self.copyDefaultWhiteChars = False
        return self

    def parseWithTabs(self):
        """Overrides default behavior to expand C{<TAB>}s to spaces before parsing the input string.
           Must be called before C{parseString} when the input grammar contains elements that
           match C{<TAB>} characters."""
        self.keepTabs = True
        return self

    def ignore(self, other):
        """Define expression to be ignored (e.g., comments) while doing pattern
           matching; may be called repeatedly, to define multiple comment or other
           ignorable patterns.
        """
        if isinstance(other, Suppress):
            if other not in self.ignoreExprs:
                self.ignoreExprs.append(other.copy())
        else:
            self.ignoreExprs.append(Suppress(other.copy()))
        return self

    def setDebugActions(self, startAction, successAction, exceptionAction):
        """Enable display of debugging messages while doing pattern matching."""
        self.debugActions = (startAction or _defaultStartDebugAction, successAction or _defaultSuccessDebugAction, exceptionAction or _defaultExceptionDebugAction)
        self.debug = True
        return self

    def setDebug(self, flag=True):
        """Enable display of debugging messages while doing pattern matching.
           Set C{flag} to True to enable, False to disable."""
        if flag:
            self.setDebugActions(_defaultStartDebugAction, _defaultSuccessDebugAction, _defaultExceptionDebugAction)
        else:
            self.debug = False
        return self

    def __str__(self):
        return self.name

    def __repr__(self):
        return _ustr(self)

    def streamline(self):
        self.streamlined = True
        self.strRepr = None
        return self

    def checkRecursion(self, parseElementList):
        pass

    def validate(self, validateTrace=[]):
        """Check defined expressions for valid structure, check for infinite recursive definitions."""
        self.checkRecursion([])

    def parseFile(self, file_or_filename, parseAll=False):
        """Execute the parse expression on the given file or filename.
           If a filename is specified (instead of a file object),
           the entire file is opened, read, and closed before parsing.
        """
        try:
            file_contents = file_or_filename.read()
        except AttributeError:
            f = open(file_or_filename, 'r')
            file_contents = f.read()
            f.close()
        try:
            return self.parseString(file_contents, parseAll)
        except ParseBaseException as exc:
            if ParserElement.verbose_stacktrace:
                raise
            else:
                raise exc

    def __eq__(self, other):
        if isinstance(other, ParserElement):
            return self is other or self.__dict__ == other.__dict__
        elif isinstance(other, basestring):
            try:
                self.parseString(_ustr(other), parseAll=True)
                return True
            except ParseBaseException:
                return False
        else:
            return super(ParserElement, self) == other

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(id(self))

    def __req__(self, other):
        return self == other

    def __rne__(self, other):
        return not self == other