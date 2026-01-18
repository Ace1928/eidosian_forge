from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class VariableParseActionArgsTest(ParseTestCase):

    def runTest(self):
        pa3 = lambda s, l, t: t
        pa2 = lambda l, t: t
        pa1 = lambda t: t
        pa0 = lambda: None

        class Callable3(object):

            def __call__(self, s, l, t):
                return t

        class Callable2(object):

            def __call__(self, l, t):
                return t

        class Callable1(object):

            def __call__(self, t):
                return t

        class Callable0(object):

            def __call__(self):
                return

        class CallableS3(object):

            def __call__(s, l, t):
                return t
            __call__ = staticmethod(__call__)

        class CallableS2(object):

            def __call__(l, t):
                return t
            __call__ = staticmethod(__call__)

        class CallableS1(object):

            def __call__(t):
                return t
            __call__ = staticmethod(__call__)

        class CallableS0(object):

            def __call__():
                return
            __call__ = staticmethod(__call__)

        class CallableC3(object):

            def __call__(cls, s, l, t):
                return t
            __call__ = classmethod(__call__)

        class CallableC2(object):

            def __call__(cls, l, t):
                return t
            __call__ = classmethod(__call__)

        class CallableC1(object):

            def __call__(cls, t):
                return t
            __call__ = classmethod(__call__)

        class CallableC0(object):

            def __call__(cls):
                return
            __call__ = classmethod(__call__)

        class parseActionHolder(object):

            def pa3(s, l, t):
                return t
            pa3 = staticmethod(pa3)

            def pa2(l, t):
                return t
            pa2 = staticmethod(pa2)

            def pa1(t):
                return t
            pa1 = staticmethod(pa1)

            def pa0():
                return
            pa0 = staticmethod(pa0)

        def paArgs(*args):
            print_(args)
            return args[2]

        class ClassAsPA0(object):

            def __init__(self):
                pass

            def __str__(self):
                return 'A'

        class ClassAsPA1(object):

            def __init__(self, t):
                print_('making a ClassAsPA1')
                self.t = t

            def __str__(self):
                return self.t[0]

        class ClassAsPA2(object):

            def __init__(self, l, t):
                self.t = t

            def __str__(self):
                return self.t[0]

        class ClassAsPA3(object):

            def __init__(self, s, l, t):
                self.t = t

            def __str__(self):
                return self.t[0]

        class ClassAsPAStarNew(tuple):

            def __new__(cls, *args):
                print_('make a ClassAsPAStarNew', args)
                return tuple.__new__(cls, *args[2].asList())

            def __str__(self):
                return ''.join(self)
        from pyparsing import Literal, OneOrMore
        A = Literal('A').setParseAction(pa0)
        B = Literal('B').setParseAction(pa1)
        C = Literal('C').setParseAction(pa2)
        D = Literal('D').setParseAction(pa3)
        E = Literal('E').setParseAction(Callable0())
        F = Literal('F').setParseAction(Callable1())
        G = Literal('G').setParseAction(Callable2())
        H = Literal('H').setParseAction(Callable3())
        I = Literal('I').setParseAction(CallableS0())
        J = Literal('J').setParseAction(CallableS1())
        K = Literal('K').setParseAction(CallableS2())
        L = Literal('L').setParseAction(CallableS3())
        M = Literal('M').setParseAction(CallableC0())
        N = Literal('N').setParseAction(CallableC1())
        O = Literal('O').setParseAction(CallableC2())
        P = Literal('P').setParseAction(CallableC3())
        Q = Literal('Q').setParseAction(paArgs)
        R = Literal('R').setParseAction(parseActionHolder.pa3)
        S = Literal('S').setParseAction(parseActionHolder.pa2)
        T = Literal('T').setParseAction(parseActionHolder.pa1)
        U = Literal('U').setParseAction(parseActionHolder.pa0)
        V = Literal('V')
        gg = OneOrMore(A | C | D | E | F | G | H | I | J | K | L | M | N | O | P | Q | R | S | U | V | B | T)
        testString = 'VUTSRQPONMLKJIHGFEDCBA'
        res = gg.parseString(testString)
        print_(res.asList())
        self.assertEqual(res.asList(), list(testString), 'Failed to parse using variable length parse actions')
        A = Literal('A').setParseAction(ClassAsPA0)
        B = Literal('B').setParseAction(ClassAsPA1)
        C = Literal('C').setParseAction(ClassAsPA2)
        D = Literal('D').setParseAction(ClassAsPA3)
        E = Literal('E').setParseAction(ClassAsPAStarNew)
        gg = OneOrMore(A | B | C | D | E | F | G | H | I | J | K | L | M | N | O | P | Q | R | S | T | U | V)
        testString = 'VUTSRQPONMLKJIHGFEDCBA'
        res = gg.parseString(testString)
        print_(list(map(str, res)))
        self.assertEqual(list(map(str, res)), list(testString), 'Failed to parse using variable length parse actions using class constructors as parse actions')