from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class MiscellaneousParserTests(ParseTestCase):

    def runTest(self):
        self.expect_warning = True
        runtests = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        if IRON_PYTHON_ENV:
            runtests = 'ABCDEGHIJKLMNOPQRSTUVWXYZ'
        if 'A' in runtests:
            print_('verify oneOf handles duplicate symbols')
            try:
                test1 = pp.oneOf('a b c d a')
            except RuntimeError:
                self.assertTrue(False, 'still have infinite loop in oneOf with duplicate symbols (string input)')
            print_('verify oneOf handles generator input')
            try:
                test1 = pp.oneOf((c for c in 'a b c d a' if not c.isspace()))
            except RuntimeError:
                self.assertTrue(False, 'still have infinite loop in oneOf with duplicate symbols (generator input)')
            print_('verify oneOf handles list input')
            try:
                test1 = pp.oneOf('a b c d a'.split())
            except RuntimeError:
                self.assertTrue(False, 'still have infinite loop in oneOf with duplicate symbols (list input)')
            print_('verify oneOf handles set input')
            try:
                test1 = pp.oneOf(set('a b c d a'))
            except RuntimeError:
                self.assertTrue(False, 'still have infinite loop in oneOf with duplicate symbols (set input)')
        if 'B' in runtests:
            print_('verify MatchFirst iterates properly')
            results = pp.quotedString.parseString("'this is a single quoted string'")
            self.assertTrue(len(results) > 0, 'MatchFirst error - not iterating over all choices')
        if 'C' in runtests:
            print_('verify proper streamline logic')
            compound = pp.Literal('A') + 'B' + 'C' + 'D'
            self.assertEqual(len(compound.exprs), 2, 'bad test setup')
            print_(compound)
            compound.streamline()
            print_(compound)
            self.assertEqual(len(compound.exprs), 4, 'streamline not working')
        if 'D' in runtests:
            print_("verify Optional's do not cause match failure if have results name")
            testGrammar = pp.Literal('A') + pp.Optional('B')('gotB') + pp.Literal('C')
            try:
                testGrammar.parseString('ABC')
                testGrammar.parseString('AC')
            except pp.ParseException as pe:
                print_(pe.pstr, '->', pe)
                self.assertTrue(False, 'error in Optional matching of string %s' % pe.pstr)
        if 'E' in runtests:
            testGrammar = pp.Literal('A') | pp.Optional('B') + pp.Literal('C') | pp.Literal('D')
            try:
                testGrammar.parseString('BC')
                testGrammar.parseString('BD')
            except pp.ParseException as pe:
                print_(pe.pstr, '->', pe)
                self.assertEqual(pe.pstr, 'BD', 'wrong test string failed to parse')
                self.assertEqual(pe.loc, 1, 'error in Optional matching, pe.loc=' + str(pe.loc))
        if 'F' in runtests:
            print_('verify behavior of validate()')

            def testValidation(grmr, gnam, isValid):
                try:
                    grmr.streamline()
                    grmr.validate()
                    self.assertTrue(isValid, 'validate() accepted invalid grammar ' + gnam)
                except pp.RecursiveGrammarException as e:
                    print_(grmr)
                    self.assertFalse(isValid, 'validate() rejected valid grammar ' + gnam)
            fwd = pp.Forward()
            g1 = pp.OneOrMore(pp.Literal('A') + 'B' + 'C' | fwd)
            g2 = pp.ZeroOrMore('C' + g1)
            fwd << pp.Group(g2)
            testValidation(fwd, 'fwd', isValid=True)
            fwd2 = pp.Forward()
            fwd2 << pp.Group('A' | fwd2)
            testValidation(fwd2, 'fwd2', isValid=False)
            fwd3 = pp.Forward()
            fwd3 << pp.Optional('A') + fwd3
            testValidation(fwd3, 'fwd3', isValid=False)
        if 'G' in runtests:
            print_('verify behavior of getName()')
            aaa = pp.Group(pp.Word('a')('A'))
            bbb = pp.Group(pp.Word('b')('B'))
            ccc = pp.Group(':' + pp.Word('c')('C'))
            g1 = 'XXX' + pp.ZeroOrMore(aaa | bbb | ccc)
            teststring = 'XXX b bb a bbb bbbb aa bbbbb :c bbbbbb aaa'
            names = []
            print_(g1.parseString(teststring).dump())
            for t in g1.parseString(teststring):
                print_(t, repr(t))
                try:
                    names.append(t[0].getName())
                except Exception:
                    try:
                        names.append(t.getName())
                    except Exception:
                        names.append(None)
            print_(teststring)
            print_(names)
            self.assertEqual(names, [None, 'B', 'B', 'A', 'B', 'B', 'A', 'B', None, 'B', 'A'], 'failure in getting names for tokens')
            from pyparsing import Keyword, Word, alphas, OneOrMore
            IF, AND, BUT = map(Keyword, 'if and but'.split())
            ident = ~(IF | AND | BUT) + Word(alphas)('non-key')
            scanner = OneOrMore(IF | AND | BUT | ident)

            def getNameTester(s, l, t):
                print_(t, t.getName())
            ident.addParseAction(getNameTester)
            scanner.parseString('lsjd sldkjf IF Saslkj AND lsdjf')
        if 'H' in runtests:
            print_('verify behavior of ParseResults.get()')
            res = sum(g1.parseString(teststring)[1:])
            print_(res.dump())
            print_(res.get('A', 'A not found'))
            print_(res.get('D', '!D'))
            self.assertEqual(res.get('A', 'A not found'), 'aaa', 'get on existing key failed')
            self.assertEqual(res.get('D', '!D'), '!D', 'get on missing key failed')
        if 'I' in runtests:
            print_("verify handling of Optional's beyond the end of string")
            testGrammar = 'A' + pp.Optional('B') + pp.Optional('C') + pp.Optional('D')
            testGrammar.parseString('A')
            testGrammar.parseString('AB')
        if 'J' in runtests:
            print_('verify non-fatal usage of Literal("")')
            e = pp.Literal('')
            try:
                e.parseString('SLJFD')
            except Exception as e:
                self.assertTrue(False, 'Failed to handle empty Literal')
        if 'K' in runtests:
            print_('verify correct line() behavior when first line is empty string')
            self.assertEqual(pp.line(0, '\nabc\ndef\n'), '', 'Error in line() with empty first line in text')
            txt = '\nabc\ndef\n'
            results = [pp.line(i, txt) for i in range(len(txt))]
            self.assertEqual(results, ['', 'abc', 'abc', 'abc', 'abc', 'def', 'def', 'def', 'def'], 'Error in line() with empty first line in text')
            txt = 'abc\ndef\n'
            results = [pp.line(i, txt) for i in range(len(txt))]
            self.assertEqual(results, ['abc', 'abc', 'abc', 'abc', 'def', 'def', 'def', 'def'], 'Error in line() with non-empty first line in text')
        if 'L' in runtests:
            print_('verify behavior with repeated tokens when packrat parsing is enabled')
            a = pp.Literal('a')
            b = pp.Literal('b')
            c = pp.Literal('c')
            abb = a + b + b
            abc = a + b + c
            aba = a + b + a
            grammar = abb | abc | aba
            self.assertEqual(''.join(grammar.parseString('aba')), 'aba', 'Packrat ABA failure!')
        if 'M' in runtests:
            print_('verify behavior of setResultsName with OneOrMore and ZeroOrMore')
            stmt = pp.Keyword('test')
            print_(pp.ZeroOrMore(stmt)('tests').parseString('test test').tests)
            print_(pp.OneOrMore(stmt)('tests').parseString('test test').tests)
            print_(pp.Optional(pp.OneOrMore(stmt)('tests')).parseString('test test').tests)
            print_(pp.Optional(pp.OneOrMore(stmt))('tests').parseString('test test').tests)
            print_(pp.Optional(pp.delimitedList(stmt))('tests').parseString('test,test').tests)
            self.assertEqual(len(pp.ZeroOrMore(stmt)('tests').parseString('test test').tests), 2, 'ZeroOrMore failure with setResultsName')
            self.assertEqual(len(pp.OneOrMore(stmt)('tests').parseString('test test').tests), 2, 'OneOrMore failure with setResultsName')
            self.assertEqual(len(pp.Optional(pp.OneOrMore(stmt)('tests')).parseString('test test').tests), 2, 'OneOrMore failure with setResultsName')
            self.assertEqual(len(pp.Optional(pp.delimitedList(stmt))('tests').parseString('test,test').tests), 2, 'delimitedList failure with setResultsName')
            self.assertEqual(len((stmt * 2)('tests').parseString('test test').tests), 2, 'multiplied(1) failure with setResultsName')
            self.assertEqual(len((stmt * (None, 2))('tests').parseString('test test').tests), 2, 'multiplied(2) failure with setResultsName')
            self.assertEqual(len((stmt * (1,))('tests').parseString('test test').tests), 2, 'multipled(3) failure with setResultsName')
            self.assertEqual(len((stmt * (2,))('tests').parseString('test test').tests), 2, 'multipled(3) failure with setResultsName')