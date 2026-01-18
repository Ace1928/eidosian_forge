from __future__ import annotations
import codecs
import os
import re
import sys
import typing
from decimal import Decimal
from typing import (
from uuid import uuid4
from rdflib.compat import long_type
from rdflib.exceptions import ParserError
from rdflib.graph import ConjunctiveGraph, Graph, QuotedGraph
from rdflib.term import (
from rdflib.parser import Parser
class SinkParser:

    def __init__(self, store: 'RDFSink', openFormula: Optional['Formula']=None, thisDoc: str='', baseURI: Optional[str]=None, genPrefix: str='', why: Optional[Callable[[], None]]=None, turtle: bool=False):
        """note: namespace names should *not* end in  # ;
        the  # will get added during qname processing"""
        self._bindings = {}
        if thisDoc != '':
            assert ':' in thisDoc, 'Document URI not absolute: <%s>' % thisDoc
            self._bindings[''] = thisDoc + '#'
        self._store = store
        if genPrefix:
            store.setGenPrefix(genPrefix)
        self._thisDoc = thisDoc
        self.lines = 0
        self.startOfLine = 0
        self._genPrefix = genPrefix
        self.keywords = ['a', 'this', 'bind', 'has', 'is', 'of', 'true', 'false']
        self.keywordsSet = 0
        self._anonymousNodes: Dict[str, BNode] = {}
        self._variables: Dict[str, Variable] = {}
        self._parentVariables: Dict[str, Variable] = {}
        self._reason = why
        self.turtle = turtle
        self.string_delimiters = ('"', "'") if turtle else ('"',)
        self._reason2: Optional[Callable[..., None]] = None
        if tracking:
            self._reason2 = BecauseOfData(store.newSymbol(thisDoc), because=self._reason)
        self._baseURI: Optional[str]
        if baseURI:
            self._baseURI = baseURI
        elif thisDoc:
            self._baseURI = thisDoc
        else:
            self._baseURI = None
        assert not self._baseURI or ':' in self._baseURI
        if not self._genPrefix:
            if self._thisDoc:
                self._genPrefix = self._thisDoc + '#_g'
            else:
                self._genPrefix = uniqueURI()
        self._formula: Optional[Formula]
        if openFormula is None and (not turtle):
            if self._thisDoc:
                self._formula = store.newFormula(thisDoc + '#_formula')
            else:
                self._formula = store.newFormula()
        else:
            self._formula = openFormula
        self._context: Optional[Formula] = self._formula
        self._parentContext: Optional[Formula] = None

    def here(self, i: int) -> str:
        """String generated from position in file

        This is for repeatability when referring people to bnodes in a document.
        This has diagnostic uses less formally, as it should point one to which
        bnode the arbitrary identifier actually is. It gives the
        line and character number of the '[' charcacter or path character
        which introduced the blank node. The first blank node is boringly
        _L1C1. It used to be used only for tracking, but for tests in general
        it makes the canonical ordering of bnodes repeatable."""
        return '%s_L%iC%i' % (self._genPrefix, self.lines, i - self.startOfLine + 1)

    def formula(self) -> Optional[Formula]:
        return self._formula

    def loadStream(self, stream: Union[IO[str], IO[bytes]]) -> Optional['Formula']:
        return self.loadBuf(stream.read())

    def loadBuf(self, buf: Union[str, bytes]) -> Optional[Formula]:
        """Parses a buffer and returns its top level formula"""
        self.startDoc()
        self.feed(buf)
        return self.endDoc()

    def feed(self, octets: Union[str, bytes]) -> None:
        """Feed an octet stream to the parser

        if BadSyntax is raised, the string
        passed in the exception object is the
        remainder after any statements have been parsed.
        So if there is more data to feed to the
        parser, it should be straightforward to recover."""
        if not isinstance(octets, str):
            s = octets.decode('utf-8')
            if len(s) > 0 and s[0] == codecs.BOM_UTF8.decode('utf-8'):
                s = s[1:]
        else:
            s = octets
        i = 0
        while i >= 0:
            j = self.skipSpace(s, i)
            if j < 0:
                return
            i = self.directiveOrStatement(s, j)
            if i < 0:
                self.BadSyntax(s, j, 'expected directive or statement')

    def directiveOrStatement(self, argstr: str, h: int) -> int:
        i = self.skipSpace(argstr, h)
        if i < 0:
            return i
        if self.turtle:
            j = self.sparqlDirective(argstr, i)
            if j >= 0:
                return j
        j = self.directive(argstr, i)
        if j >= 0:
            return self.checkDot(argstr, j)
        j = self.statement(argstr, i)
        if j >= 0:
            return self.checkDot(argstr, j)
        return j

    def tok(self, tok: str, argstr: str, i: int, colon: bool=False) -> int:
        """Check for keyword.  Space must have been stripped on entry and
        we must not be at end of file.

        if colon, then keyword followed by colon is ok
        (@prefix:<blah> is ok, rdf:type shortcut a must be followed by ws)
        """
        assert tok[0] not in _notNameChars
        if argstr[i] == '@':
            i += 1
        elif tok not in self.keywords:
            return -1
        i_plus_len_tok = i + len(tok)
        if argstr[i:i_plus_len_tok] == tok and argstr[i_plus_len_tok] in _notKeywordsChars or (colon and argstr[i_plus_len_tok] == ':'):
            return i_plus_len_tok
        else:
            return -1

    def sparqlTok(self, tok: str, argstr: str, i: int) -> int:
        """Check for SPARQL keyword.  Space must have been stripped on entry
        and we must not be at end of file.
        Case insensitive and not preceded by @
        """
        assert tok[0] not in _notNameChars
        len_tok = len(tok)
        if argstr[i:i + len_tok].lower() == tok.lower() and argstr[i + len_tok] in _notQNameChars:
            i += len_tok
            return i
        else:
            return -1

    def directive(self, argstr: str, i: int) -> int:
        j = self.skipSpace(argstr, i)
        if j < 0:
            return j
        res: typing.List[str] = []
        j = self.tok('bind', argstr, i)
        if j > 0:
            self.BadSyntax(argstr, i, 'keyword bind is obsolete: use @prefix')
        j = self.tok('keywords', argstr, i)
        if j > 0:
            if self.turtle:
                self.BadSyntax(argstr, i, "Found 'keywords' when in Turtle mode.")
            i = self.commaSeparatedList(argstr, j, res, self.bareWord)
            if i < 0:
                self.BadSyntax(argstr, i, "'@keywords' needs comma separated list of words")
            self.setKeywords(res[:])
            return i
        j = self.tok('forAll', argstr, i)
        if j > 0:
            if self.turtle:
                self.BadSyntax(argstr, i, "Found 'forAll' when in Turtle mode.")
            i = self.commaSeparatedList(argstr, j, res, self.uri_ref2)
            if i < 0:
                self.BadSyntax(argstr, i, 'Bad variable list after @forAll')
            for x in res:
                if x not in self._variables or x in self._parentVariables:
                    self._variables[x] = self._context.newUniversal(x)
            return i
        j = self.tok('forSome', argstr, i)
        if j > 0:
            if self.turtle:
                self.BadSyntax(argstr, i, "Found 'forSome' when in Turtle mode.")
            i = self.commaSeparatedList(argstr, j, res, self.uri_ref2)
            if i < 0:
                self.BadSyntax(argstr, i, 'Bad variable list after @forSome')
            for x in res:
                self._context.declareExistential(x)
            return i
        j = self.tok('prefix', argstr, i, colon=True)
        if j >= 0:
            t: typing.List[Union[Identifier, Tuple[str, str]]] = []
            i = self.qname(argstr, j, t)
            if i < 0:
                self.BadSyntax(argstr, j, 'expected qname after @prefix')
            j = self.uri_ref2(argstr, i, t)
            if j < 0:
                self.BadSyntax(argstr, i, 'expected <uriref> after @prefix _qname_')
            ns: str = self.uriOf(t[1])
            if self._baseURI:
                ns = join(self._baseURI, ns)
            elif ':' not in ns:
                self.BadSyntax(argstr, j, f'With no base URI, cannot use relative URI in @prefix <{ns}>')
            assert ':' in ns
            self._bindings[t[0][0]] = ns
            self.bind(t[0][0], hexify(ns))
            return j
        j = self.tok('base', argstr, i)
        if j >= 0:
            t = []
            i = self.uri_ref2(argstr, j, t)
            if i < 0:
                self.BadSyntax(argstr, j, 'expected <uri> after @base ')
            ns = self.uriOf(t[0])
            if self._baseURI:
                ns = join(self._baseURI, ns)
            else:
                self.BadSyntax(argstr, j, 'With no previous base URI, cannot use ' + 'relative URI in @base  <' + ns + '>')
            assert ':' in ns
            self._baseURI = ns
            return i
        return -1

    def sparqlDirective(self, argstr: str, i: int) -> int:
        """
        turtle and trig support BASE/PREFIX without @ and without
        terminating .
        """
        j = self.skipSpace(argstr, i)
        if j < 0:
            return j
        j = self.sparqlTok('PREFIX', argstr, i)
        if j >= 0:
            t: typing.List[Any] = []
            i = self.qname(argstr, j, t)
            if i < 0:
                self.BadSyntax(argstr, j, 'expected qname after @prefix')
            j = self.uri_ref2(argstr, i, t)
            if j < 0:
                self.BadSyntax(argstr, i, 'expected <uriref> after @prefix _qname_')
            ns = self.uriOf(t[1])
            if self._baseURI:
                ns = join(self._baseURI, ns)
            elif ':' not in ns:
                self.BadSyntax(argstr, j, 'With no base URI, cannot use ' + 'relative URI in @prefix <' + ns + '>')
            assert ':' in ns
            self._bindings[t[0][0]] = ns
            self.bind(t[0][0], hexify(ns))
            return j
        j = self.sparqlTok('BASE', argstr, i)
        if j >= 0:
            t = []
            i = self.uri_ref2(argstr, j, t)
            if i < 0:
                self.BadSyntax(argstr, j, 'expected <uri> after @base ')
            ns = self.uriOf(t[0])
            if self._baseURI:
                ns = join(self._baseURI, ns)
            else:
                self.BadSyntax(argstr, j, 'With no previous base URI, cannot use ' + 'relative URI in @base  <' + ns + '>')
            assert ':' in ns
            self._baseURI = ns
            return i
        return -1

    def bind(self, qn: str, uri: bytes) -> None:
        assert isinstance(uri, bytes), 'Any unicode must be %x-encoded already'
        if qn == '':
            self._store.setDefaultNamespace(uri)
        else:
            self._store.bind(qn, uri)

    def setKeywords(self, k: Optional[typing.List[str]]) -> None:
        """Takes a list of strings"""
        if k is None:
            self.keywordsSet = 0
        else:
            self.keywords = k
            self.keywordsSet = 1

    def startDoc(self) -> None:
        self._store.startDoc(self._formula)

    def endDoc(self) -> Optional['Formula']:
        """Signal end of document and stop parsing. returns formula"""
        self._store.endDoc(self._formula)
        return self._formula

    def makeStatement(self, quadruple) -> None:
        self._store.makeStatement(quadruple, why=self._reason2)

    def statement(self, argstr: str, i: int) -> int:
        r: typing.List[Any] = []
        i = self.object(argstr, i, r)
        if i < 0:
            return i
        j = self.property_list(argstr, i, r[0])
        if j < 0:
            self.BadSyntax(argstr, i, 'expected propertylist')
        return j

    def subject(self, argstr: str, i: int, res: MutableSequence[Any]) -> int:
        return self.item(argstr, i, res)

    def verb(self, argstr: str, i: int, res: MutableSequence[Any]) -> int:
        """has _prop_
        is _prop_ of
        a
        =
        _prop_
        >- prop ->
        <- prop -<
        _operator_"""
        j = self.skipSpace(argstr, i)
        if j < 0:
            return j
        r: typing.List[Any] = []
        j = self.tok('has', argstr, i)
        if j >= 0:
            if self.turtle:
                self.BadSyntax(argstr, i, "Found 'has' keyword in Turtle mode")
            i = self.prop(argstr, j, r)
            if i < 0:
                self.BadSyntax(argstr, j, "expected property after 'has'")
            res.append(('->', r[0]))
            return i
        j = self.tok('is', argstr, i)
        if j >= 0:
            if self.turtle:
                self.BadSyntax(argstr, i, "Found 'is' keyword in Turtle mode")
            i = self.prop(argstr, j, r)
            if i < 0:
                self.BadSyntax(argstr, j, "expected <property> after 'is'")
            j = self.skipSpace(argstr, i)
            if j < 0:
                self.BadSyntax(argstr, i, "End of file found, expected property after 'is'")
            i = j
            j = self.tok('of', argstr, i)
            if j < 0:
                self.BadSyntax(argstr, i, "expected 'of' after 'is' <prop>")
            res.append(('<-', r[0]))
            return j
        j = self.tok('a', argstr, i)
        if j >= 0:
            res.append(('->', RDF_type))
            return j
        if argstr[i:i + 2] == '<=':
            if self.turtle:
                self.BadSyntax(argstr, i, "Found '<=' in Turtle mode. ")
            res.append(('<-', self._store.newSymbol(Logic_NS + 'implies')))
            return i + 2
        if argstr[i] == '=':
            if self.turtle:
                self.BadSyntax(argstr, i, "Found '=' in Turtle mode")
            if argstr[i + 1] == '>':
                res.append(('->', self._store.newSymbol(Logic_NS + 'implies')))
                return i + 2
            res.append(('->', DAML_sameAs))
            return i + 1
        if argstr[i:i + 2] == ':=':
            if self.turtle:
                self.BadSyntax(argstr, i, "Found ':=' in Turtle mode")
            res.append(('->', Logic_NS + 'becomes'))
            return i + 2
        j = self.prop(argstr, i, r)
        if j >= 0:
            res.append(('->', r[0]))
            return j
        if argstr[i:i + 2] == '>-' or argstr[i:i + 2] == '<-':
            self.BadSyntax(argstr, j, '>- ... -> syntax is obsolete.')
        return -1

    def prop(self, argstr: str, i: int, res: MutableSequence[Any]) -> int:
        return self.item(argstr, i, res)

    def item(self, argstr: str, i, res: MutableSequence[Any]) -> int:
        return self.path(argstr, i, res)

    def blankNode(self, uri: Optional[str]=None) -> BNode:
        return self._store.newBlankNode(self._context, uri, why=self._reason2)

    def path(self, argstr: str, i: int, res: MutableSequence[Any]) -> int:
        """Parse the path production."""
        j = self.nodeOrLiteral(argstr, i, res)
        if j < 0:
            return j
        while argstr[j] in {'!', '^'}:
            ch = argstr[j]
            subj = res.pop()
            obj = self.blankNode(uri=self.here(j))
            j = self.node(argstr, j + 1, res)
            if j < 0:
                self.BadSyntax(argstr, j, 'EOF found in middle of path syntax')
            pred = res.pop()
            if ch == '^':
                self.makeStatement((self._context, pred, obj, subj))
            else:
                self.makeStatement((self._context, pred, subj, obj))
            res.append(obj)
        return j

    def anonymousNode(self, ln: str) -> BNode:
        """Remember or generate a term for one of these _: anonymous nodes"""
        term = self._anonymousNodes.get(ln, None)
        if term is not None:
            return term
        term = self._store.newBlankNode(self._context, why=self._reason2)
        self._anonymousNodes[ln] = term
        return term

    def node(self, argstr: str, i: int, res: MutableSequence[Any], subjectAlready: Optional[Node]=None) -> int:
        """Parse the <node> production.
        Space is now skipped once at the beginning
        instead of in multiple calls to self.skipSpace().
        """
        subj: Optional[Node] = subjectAlready
        j = self.skipSpace(argstr, i)
        if j < 0:
            return j
        i = j
        ch = argstr[i]
        if ch == '[':
            bnodeID = self.here(i)
            j = self.skipSpace(argstr, i + 1)
            if j < 0:
                self.BadSyntax(argstr, i, "EOF after '['")
            if argstr[j] == '=':
                if self.turtle:
                    self.BadSyntax(argstr, j, "Found '[=' or '[ =' when in turtle mode.")
                i = j + 1
                objs: typing.List[Node] = []
                j = self.objectList(argstr, i, objs)
                if j >= 0:
                    subj = objs[0]
                    if len(objs) > 1:
                        for obj in objs:
                            self.makeStatement((self._context, DAML_sameAs, subj, obj))
                    j = self.skipSpace(argstr, j)
                    if j < 0:
                        self.BadSyntax(argstr, i, 'EOF when objectList expected after [ = ')
                    if argstr[j] == ';':
                        j += 1
                else:
                    self.BadSyntax(argstr, i, 'objectList expected after [= ')
            if subj is None:
                subj = self.blankNode(uri=bnodeID)
            i = self.property_list(argstr, j, subj)
            if i < 0:
                self.BadSyntax(argstr, j, 'property_list expected')
            j = self.skipSpace(argstr, i)
            if j < 0:
                self.BadSyntax(argstr, i, "EOF when ']' expected after [ <propertyList>")
            if argstr[j] != ']':
                self.BadSyntax(argstr, j, "']' expected")
            res.append(subj)
            return j + 1
        if not self.turtle and ch == '{':
            ch2 = argstr[i + 1]
            if ch2 == '$':
                i += 1
                j = i + 1
                List = []
                first_run = True
                while 1:
                    i = self.skipSpace(argstr, j)
                    if i < 0:
                        self.BadSyntax(argstr, i, "needed '$}', found end.")
                    if argstr[i:i + 2] == '$}':
                        j = i + 2
                        break
                    if not first_run:
                        if argstr[i] == ',':
                            i += 1
                        else:
                            self.BadSyntax(argstr, i, "expected: ','")
                    else:
                        first_run = False
                    item: typing.List[Any] = []
                    j = self.item(argstr, i, item)
                    if j < 0:
                        self.BadSyntax(argstr, i, "expected item in set or '$}'")
                    List.append(self._store.intern(item[0]))
                res.append(self._store.newSet(List, self._context))
                return j
            else:
                j = i + 1
                oldParentContext = self._parentContext
                self._parentContext = self._context
                parentAnonymousNodes = self._anonymousNodes
                grandParentVariables = self._parentVariables
                self._parentVariables = self._variables
                self._anonymousNodes = {}
                self._variables = self._variables.copy()
                reason2 = self._reason2
                self._reason2 = becauseSubexpression
                if subj is None:
                    subj = self._store.newFormula()
                self._context = subj
                while 1:
                    i = self.skipSpace(argstr, j)
                    if i < 0:
                        self.BadSyntax(argstr, i, "needed '}', found end.")
                    if argstr[i] == '}':
                        j = i + 1
                        break
                    j = self.directiveOrStatement(argstr, i)
                    if j < 0:
                        self.BadSyntax(argstr, i, "expected statement or '}'")
                self._anonymousNodes = parentAnonymousNodes
                self._variables = self._parentVariables
                self._parentVariables = grandParentVariables
                self._context = self._parentContext
                self._reason2 = reason2
                self._parentContext = oldParentContext
                res.append(subj.close())
                return j
        if ch == '(':
            thing_type: Callable[[typing.List[Any], Optional[Formula]], Union[Set[Any], IdentifiedNode]]
            thing_type = self._store.newList
            ch2 = argstr[i + 1]
            if ch2 == '$':
                thing_type = self._store.newSet
                i += 1
            j = i + 1
            List = []
            while 1:
                i = self.skipSpace(argstr, j)
                if i < 0:
                    self.BadSyntax(argstr, i, "needed ')', found end.")
                if argstr[i] == ')':
                    j = i + 1
                    break
                item = []
                j = self.item(argstr, i, item)
                if j < 0:
                    self.BadSyntax(argstr, i, "expected item in list or ')'")
                List.append(self._store.intern(item[0]))
            res.append(thing_type(List, self._context))
            return j
        j = self.tok('this', argstr, i)
        if j >= 0:
            self.BadSyntax(argstr, i, "Keyword 'this' was ancient N3. Now use " + '@forSome and @forAll keywords.')
        j = self.tok('true', argstr, i)
        if j >= 0:
            res.append(True)
            return j
        j = self.tok('false', argstr, i)
        if j >= 0:
            res.append(False)
            return j
        if subj is None:
            j = self.uri_ref2(argstr, i, res)
            if j >= 0:
                return j
        return -1

    def property_list(self, argstr: str, i: int, subj: Node) -> int:
        """Parse property list
        Leaves the terminating punctuation in the buffer
        """
        while 1:
            while 1:
                j = self.skipSpace(argstr, i)
                if j < 0:
                    self.BadSyntax(argstr, i, 'EOF found when expected verb in property list')
                if argstr[j] != ';':
                    break
                i = j + 1
            if argstr[j:j + 2] == ':-':
                if self.turtle:
                    self.BadSyntax(argstr, j, "Found in ':-' in Turtle mode")
                i = j + 2
                res: typing.List[Any] = []
                j = self.node(argstr, i, res, subj)
                if j < 0:
                    self.BadSyntax(argstr, i, 'bad {} or () or [] node after :- ')
                i = j
                continue
            i = j
            v: typing.List[Any] = []
            j = self.verb(argstr, i, v)
            if j <= 0:
                return i
            objs: typing.List[Any] = []
            i = self.objectList(argstr, j, objs)
            if i < 0:
                self.BadSyntax(argstr, j, 'objectList expected')
            for obj in objs:
                dira, sym = v[0]
                if dira == '->':
                    self.makeStatement((self._context, sym, subj, obj))
                else:
                    self.makeStatement((self._context, sym, obj, subj))
            j = self.skipSpace(argstr, i)
            if j < 0:
                self.BadSyntax(argstr, j, 'EOF found in list of objects')
            if argstr[i] != ';':
                return i
            i += 1

    def commaSeparatedList(self, argstr: str, j: int, res: MutableSequence[Any], what: Callable[[str, int, MutableSequence[Any]], int]) -> int:
        """return value: -1 bad syntax; >1 new position in argstr
        res has things found appended
        """
        i = self.skipSpace(argstr, j)
        if i < 0:
            self.BadSyntax(argstr, i, 'EOF found expecting comma sep list')
        if argstr[i] == '.':
            return j
        i = what(argstr, i, res)
        if i < 0:
            return -1
        while 1:
            j = self.skipSpace(argstr, i)
            if j < 0:
                return j
            ch = argstr[j]
            if ch != ',':
                if ch != '.':
                    return -1
                return j
            i = what(argstr, j + 1, res)
            if i < 0:
                self.BadSyntax(argstr, i, 'bad list content')

    def objectList(self, argstr: str, i: int, res: MutableSequence[Any]) -> int:
        i = self.object(argstr, i, res)
        if i < 0:
            return -1
        while 1:
            j = self.skipSpace(argstr, i)
            if j < 0:
                self.BadSyntax(argstr, j, 'EOF found after object')
            if argstr[j] != ',':
                return j
            i = self.object(argstr, j + 1, res)
            if i < 0:
                return i

    def checkDot(self, argstr: str, i: int) -> int:
        j = self.skipSpace(argstr, i)
        if j < 0:
            return j
        ch = argstr[j]
        if ch == '.':
            return j + 1
        if ch == '}':
            return j
        if ch == ']':
            return j
        self.BadSyntax(argstr, j, "expected '.' or '}' or ']' at end of statement")

    def uri_ref2(self, argstr: str, i: int, res: MutableSequence[Any]) -> int:
        """Generate uri from n3 representation.

        Note that the RDF convention of directly concatenating
        NS and local name is now used though I prefer inserting a '#'
        to make the namesapces look more like what XML folks expect.
        """
        qn: typing.List[Any] = []
        j = self.qname(argstr, i, qn)
        if j >= 0:
            pfx, ln = qn[0]
            if pfx is None:
                assert 0, 'not used?'
                ns = self._baseURI + ADDED_HASH
            else:
                try:
                    ns = self._bindings[pfx]
                except KeyError:
                    if pfx == '_':
                        res.append(self.anonymousNode(ln))
                        return j
                    if not self.turtle and pfx == '':
                        ns = join(self._baseURI or '', '#')
                    else:
                        self.BadSyntax(argstr, i, 'Prefix "%s:" not bound' % pfx)
            symb = self._store.newSymbol(ns + ln)
            res.append(self._variables.get(symb, symb))
            return j
        i = self.skipSpace(argstr, i)
        if i < 0:
            return -1
        if argstr[i] == '?':
            v: typing.List[Any] = []
            j = self.variable(argstr, i, v)
            if j > 0:
                res.append(v[0])
                return j
            return -1
        elif argstr[i] == '<':
            st = i + 1
            i = argstr.find('>', st)
            if i >= 0:
                uref = argstr[st:i]
                uref = unicodeEscape8.sub(unicodeExpand, uref)
                uref = unicodeEscape4.sub(unicodeExpand, uref)
                if self._baseURI:
                    uref = join(self._baseURI, uref)
                else:
                    assert ':' in uref, 'With no base URI, cannot deal with relative URIs'
                if argstr[i - 1] == '#' and (not uref[-1:] == '#'):
                    uref += '#'
                symb = self._store.newSymbol(uref)
                res.append(self._variables.get(symb, symb))
                return i + 1
            self.BadSyntax(argstr, j, 'unterminated URI reference')
        elif self.keywordsSet:
            v = []
            j = self.bareWord(argstr, i, v)
            if j < 0:
                return -1
            if v[0] in self.keywords:
                self.BadSyntax(argstr, i, 'Keyword "%s" not allowed here.' % v[0])
            res.append(self._store.newSymbol(self._bindings[''] + v[0]))
            return j
        else:
            return -1

    def skipSpace(self, argstr: str, i: int) -> int:
        """Skip white space, newlines and comments.
        return -1 if EOF, else position of first non-ws character"""
        try:
            while True:
                ch = argstr[i]
                if ch in {' ', '\t'}:
                    i += 1
                    continue
                elif ch not in {'#', '\r', '\n'}:
                    return i
                break
        except IndexError:
            return -1
        while 1:
            m = eol.match(argstr, i)
            if m is None:
                break
            self.lines += 1
            self.startOfLine = i = m.end()
        m = ws.match(argstr, i)
        if m is not None:
            i = m.end()
        m = eof.match(argstr, i)
        return i if m is None else -1

    def variable(self, argstr: str, i: int, res) -> int:
        """?abc -> variable(:abc)"""
        j = self.skipSpace(argstr, i)
        if j < 0:
            return -1
        if argstr[j] != '?':
            return -1
        j += 1
        i = j
        if argstr[j] in numberChars:
            self.BadSyntax(argstr, j, "Variable name can't start with '%s'" % argstr[j])
        len_argstr = len(argstr)
        while i < len_argstr and argstr[i] not in _notKeywordsChars:
            i += 1
        if self._parentContext is None:
            varURI = self._store.newSymbol(self._baseURI + '#' + argstr[j:i])
            if varURI not in self._variables:
                self._variables[varURI] = self._context.newUniversal(varURI, why=self._reason2)
            res.append(self._variables[varURI])
            return i
        varURI = self._store.newSymbol(self._baseURI + '#' + argstr[j:i])
        if varURI not in self._parentVariables:
            self._parentVariables[varURI] = self._parentContext.newUniversal(varURI, why=self._reason2)
        res.append(self._parentVariables[varURI])
        return i

    def bareWord(self, argstr: str, i: int, res: MutableSequence[Any]) -> int:
        """abc -> :abc"""
        j = self.skipSpace(argstr, i)
        if j < 0:
            return -1
        if argstr[j] in numberChars or argstr[j] in _notKeywordsChars:
            return -1
        i = j
        len_argstr = len(argstr)
        while i < len_argstr and argstr[i] not in _notKeywordsChars:
            i += 1
        res.append(argstr[j:i])
        return i

    def qname(self, argstr: str, i: int, res: MutableSequence[Union[Identifier, Tuple[str, str]]]) -> int:
        """
        xyz:def -> ('xyz', 'def')
        If not in keywords and keywordsSet: def -> ('', 'def')
        :def -> ('', 'def')
        """
        i = self.skipSpace(argstr, i)
        if i < 0:
            return -1
        c = argstr[i]
        if c in numberCharsPlus:
            return -1
        len_argstr = len(argstr)
        if c not in _notNameChars:
            j = i
            i += 1
            try:
                while argstr[i] not in _notNameChars:
                    i += 1
            except IndexError:
                pass
            if argstr[i - 1] == '.':
                i -= 1
                if i == j:
                    return -1
            ln = argstr[j:i]
        else:
            ln = ''
        if i < len_argstr and argstr[i] == ':':
            pfx = ln
            if pfx == '_':
                allowedChars = _notNameChars
            else:
                allowedChars = _notQNameChars
            i += 1
            lastslash = False
            start = i
            ln = ''
            while i < len_argstr:
                c = argstr[i]
                if c == '\\' and (not lastslash):
                    lastslash = True
                    if start < i:
                        ln += argstr[start:i]
                    start = i + 1
                elif c not in allowedChars or lastslash:
                    if lastslash:
                        if c not in escapeChars:
                            raise BadSyntax(self._thisDoc, self.lines, argstr, i, 'illegal escape ' + c)
                    elif c == '%':
                        if argstr[i + 1] not in hexChars or argstr[i + 2] not in hexChars:
                            raise BadSyntax(self._thisDoc, self.lines, argstr, i, 'illegal hex escape ' + c)
                    lastslash = False
                else:
                    break
                i += 1
            if lastslash:
                raise BadSyntax(self._thisDoc, self.lines, argstr, i, 'qname cannot end with \\')
            if argstr[i - 1] == '.':
                if len(ln) == 0 and start == i:
                    return -1
                i -= 1
            if start < i:
                ln += argstr[start:i]
            res.append((pfx, ln))
            return i
        else:
            if ln and self.keywordsSet and (ln not in self.keywords):
                res.append(('', ln))
                return i
            return -1

    def object(self, argstr: str, i: int, res: MutableSequence[Any]) -> int:
        j = self.subject(argstr, i, res)
        if j >= 0:
            return j
        else:
            j = self.skipSpace(argstr, i)
            if j < 0:
                return -1
            else:
                i = j
            ch = argstr[i]
            if ch in self.string_delimiters:
                ch_three = ch * 3
                if argstr[i:i + 3] == ch_three:
                    delim = ch_three
                    i += 3
                else:
                    delim = ch
                    i += 1
                j, s = self.strconst(argstr, i, delim)
                res.append(self._store.newLiteral(s))
                return j
            else:
                return -1

    def nodeOrLiteral(self, argstr: str, i: int, res: MutableSequence[Any]) -> int:
        j = self.node(argstr, i, res)
        startline = self.lines
        if j >= 0:
            return j
        else:
            j = self.skipSpace(argstr, i)
            if j < 0:
                return -1
            else:
                i = j
            ch = argstr[i]
            if ch in numberCharsPlus:
                m = exponent_syntax.match(argstr, i)
                if m:
                    j = m.end()
                    res.append(float(argstr[i:j]))
                    return j
                m = decimal_syntax.match(argstr, i)
                if m:
                    j = m.end()
                    res.append(Decimal(argstr[i:j]))
                    return j
                m = integer_syntax.match(argstr, i)
                if m:
                    j = m.end()
                    res.append(long_type(argstr[i:j]))
                    return j
            ch_three = ch * 3
            if ch in self.string_delimiters:
                if argstr[i:i + 3] == ch_three:
                    delim = ch_three
                    i += 3
                else:
                    delim = ch
                    i += 1
                dt = None
                j, s = self.strconst(argstr, i, delim)
                lang = None
                if argstr[j] == '@':
                    m = langcode.match(argstr, j + 1)
                    if m is None:
                        raise BadSyntax(self._thisDoc, startline, argstr, i, 'Bad language code syntax on string ' + 'literal, after @')
                    i = m.end()
                    lang = argstr[j + 1:i]
                    j = i
                if argstr[j:j + 2] == '^^':
                    res2: typing.List[Any] = []
                    j = self.uri_ref2(argstr, j + 2, res2)
                    dt = res2[0]
                res.append(self._store.newLiteral(s, dt, lang))
                return j
            else:
                return -1

    def uriOf(self, sym: Union[Identifier, Tuple[str, str]]) -> str:
        if isinstance(sym, tuple):
            return sym[1]
        return sym

    def strconst(self, argstr: str, i: int, delim: str) -> Tuple[int, str]:
        """parse an N3 string constant delimited by delim.
        return index, val
        """
        delim1 = delim[0]
        delim2, delim3, delim4, delim5 = (delim1 * 2, delim1 * 3, delim1 * 4, delim1 * 5)
        j = i
        ustr = ''
        startline = self.lines
        len_argstr = len(argstr)
        while j < len_argstr:
            if argstr[j] == delim1:
                if delim == delim1:
                    i = j + 1
                    return (i, ustr)
                if delim == delim3:
                    if argstr[j:j + 5] == delim5:
                        i = j + 5
                        ustr += delim2
                        return (i, ustr)
                    if argstr[j:j + 4] == delim4:
                        i = j + 4
                        ustr += delim1
                        return (i, ustr)
                    if argstr[j:j + 3] == delim3:
                        i = j + 3
                        return (i, ustr)
                    j += 1
                    ustr += delim1
                    continue
            m = interesting.search(argstr, j)
            assert m, 'Quote expected in string at ^ in %s^%s' % (argstr[j - 20:j], argstr[j:j + 20])
            i = m.start()
            try:
                ustr += argstr[j:i]
            except UnicodeError:
                err = ''
                for c in argstr[j:i]:
                    err = err + ' %02x' % ord(c)
                streason = sys.exc_info()[1].__str__()
                raise BadSyntax(self._thisDoc, startline, argstr, j, 'Unicode error appending characters' + ' %s to string, because\n\t%s' % (err, streason))
            ch = argstr[i]
            if ch == delim1:
                j = i
                continue
            elif ch in {'"', "'"} and ch != delim1:
                ustr += ch
                j = i + 1
                continue
            elif ch in {'\r', '\n'}:
                if delim == delim1:
                    raise BadSyntax(self._thisDoc, startline, argstr, i, 'newline found in string literal')
                self.lines += 1
                ustr += ch
                j = i + 1
                self.startOfLine = j
            elif ch == '\\':
                j = i + 1
                ch = argstr[j]
                if not ch:
                    raise BadSyntax(self._thisDoc, startline, argstr, i, 'unterminated string literal (2)')
                k = 'abfrtvn\\"\''.find(ch)
                if k >= 0:
                    uch = '\x07\x08\x0c\r\t\x0b\n\\"\''[k]
                    ustr += uch
                    j += 1
                elif ch == 'u':
                    j, ch = self.uEscape(argstr, j + 1, startline)
                    ustr += ch
                elif ch == 'U':
                    j, ch = self.UEscape(argstr, j + 1, startline)
                    ustr += ch
                else:
                    self.BadSyntax(argstr, i, 'bad escape')
        self.BadSyntax(argstr, i, 'unterminated string literal')

    def _unicodeEscape(self, argstr: str, i: int, startline: int, reg: Pattern[str], n: int, prefix: str) -> Tuple[int, str]:
        if len(argstr) < i + n:
            raise BadSyntax(self._thisDoc, startline, argstr, i, 'unterminated string literal(3)')
        try:
            return (i + n, reg.sub(unicodeExpand, '\\' + prefix + argstr[i:i + n]))
        except Exception:
            raise BadSyntax(self._thisDoc, startline, argstr, i, 'bad string literal hex escape: ' + argstr[i:i + n])

    def uEscape(self, argstr: str, i: int, startline: int) -> Tuple[int, str]:
        return self._unicodeEscape(argstr, i, startline, unicodeEscape4, 4, 'u')

    def UEscape(self, argstr: str, i: int, startline: int) -> Tuple[int, str]:
        return self._unicodeEscape(argstr, i, startline, unicodeEscape8, 8, 'U')

    def BadSyntax(self, argstr: str, i: int, msg: str) -> NoReturn:
        raise BadSyntax(self._thisDoc, self.lines, argstr, i, msg)