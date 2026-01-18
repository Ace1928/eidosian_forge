from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ParseResultsPickleTest(ParseTestCase):

    def runTest(self):
        from pyparsing import makeHTMLTags, ParseResults
        import pickle
        body = makeHTMLTags('BODY')[0]
        result = body.parseString("<BODY BGCOLOR='#00FFBB' FGCOLOR=black>")
        if VERBOSE:
            print_(result.dump())
            print_()
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            print_('Test pickle dump protocol', protocol)
            try:
                pickleString = pickle.dumps(result, protocol)
            except Exception as e:
                print_('dumps exception:', e)
                newresult = ParseResults()
            else:
                newresult = pickle.loads(pickleString)
                if VERBOSE:
                    print_(newresult.dump())
                    print_()
            self.assertEqual(result.dump(), newresult.dump(), 'Error pickling ParseResults object (protocol=%d)' % protocol)
        import pyparsing as pp
        word = pp.Word(pp.alphas + "'.")
        salutation = pp.OneOrMore(word)
        comma = pp.Literal(',')
        greetee = pp.OneOrMore(word)
        endpunc = pp.oneOf('! ?')
        greeting = salutation + pp.Suppress(comma) + greetee + pp.Suppress(endpunc)
        greeting.setParseAction(PickleTest_Greeting)
        string = 'Good morning, Miss Crabtree!'
        result = greeting.parseString(string)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            print_('Test pickle dump protocol', protocol)
            try:
                pickleString = pickle.dumps(result, protocol)
            except Exception as e:
                print_('dumps exception:', e)
                newresult = ParseResults()
            else:
                newresult = pickle.loads(pickleString)
            print_(newresult.dump())
            self.assertEqual(newresult.dump(), result.dump(), 'failed to pickle/unpickle ParseResults: expected %r, got %r' % (result, newresult))