from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class UpcaseDowncaseUnicode(ParseTestCase):

    def runTest(self):
        import pyparsing as pp
        from pyparsing import pyparsing_unicode as ppu
        import sys
        if PY_3:
            unichr = chr
        else:
            from __builtin__ import unichr
        a = u'¿Cómo esta usted?'
        if not JYTHON_ENV:
            ualphas = ppu.alphas
        else:
            ualphas = ''.join((unichr(i) for i in list(range(55296)) + list(range(57344, sys.maxunicode)) if unichr(i).isalpha()))
        uword = pp.Word(ualphas).setParseAction(pp.upcaseTokens)
        print_ = lambda *args: None
        print_(uword.searchString(a))
        uword = pp.Word(ualphas).setParseAction(pp.downcaseTokens)
        print_(uword.searchString(a))
        kw = pp.Keyword('mykey', caseless=True).setParseAction(pp.upcaseTokens)('rname')
        ret = kw.parseString('mykey')
        print_(ret.rname)
        self.assertEqual(ret.rname, 'MYKEY', 'failed to upcase with named result')
        kw = pp.Keyword('mykey', caseless=True).setParseAction(pp.pyparsing_common.upcaseTokens)('rname')
        ret = kw.parseString('mykey')
        print_(ret.rname)
        self.assertEqual(ret.rname, 'MYKEY', 'failed to upcase with named result (pyparsing_common)')
        kw = pp.Keyword('MYKEY', caseless=True).setParseAction(pp.pyparsing_common.downcaseTokens)('rname')
        ret = kw.parseString('mykey')
        print_(ret.rname)
        self.assertEqual(ret.rname, 'mykey', 'failed to upcase with named result')
        if not IRON_PYTHON_ENV:
            html = u'<TR class=maintxt bgColor=#ffffff>                 <TD vAlign=top>Производитель, модель</TD>                 <TD vAlign=top><STRONG>BenQ-Siemens CF61</STRONG></TD>             '
            text_manuf = u'Производитель, модель'
            manufacturer = pp.Literal(text_manuf)
            td_start, td_end = pp.makeHTMLTags('td')
            manuf_body = td_start.suppress() + manufacturer + pp.SkipTo(td_end)('cells*') + td_end.suppress()