from __future__ import annotations
import os
import sys
import unittest
import textwrap
from typing import Any
from . import markdown, Markdown, util
class LegacyTestMeta(type):

    def __new__(cls, name, bases, dct):

        def generate_test(infile, outfile, normalize, kwargs):

            def test(self):
                with open(infile, encoding='utf-8') as f:
                    input = f.read()
                with open(outfile, encoding='utf-8') as f:
                    expected = f.read().replace('\r\n', '\n')
                output = markdown(input, **kwargs)
                if tidylib and normalize:
                    try:
                        expected = _normalize_whitespace(expected)
                        output = _normalize_whitespace(output)
                    except OSError:
                        self.skipTest("Tidylib's c library not available.")
                elif normalize:
                    self.skipTest('Tidylib not available.')
                self.assertMultiLineEqual(output, expected)
            return test
        location = dct.get('location', '')
        exclude = dct.get('exclude', [])
        normalize = dct.get('normalize', False)
        input_ext = dct.get('input_ext', '.txt')
        output_ext = dct.get('output_ext', '.html')
        kwargs = dct.get('default_kwargs', Kwargs())
        if os.path.isdir(location):
            for file in os.listdir(location):
                infile = os.path.join(location, file)
                if os.path.isfile(infile):
                    tname, ext = os.path.splitext(file)
                    if ext == input_ext:
                        outfile = os.path.join(location, tname + output_ext)
                        tname = tname.replace(' ', '_').replace('-', '_')
                        kws = kwargs.copy()
                        if tname in dct:
                            kws.update(dct[tname])
                        test_name = 'test_%s' % tname
                        if tname not in exclude:
                            dct[test_name] = generate_test(infile, outfile, normalize, kws)
                        else:
                            dct[test_name] = unittest.skip('Excluded')(lambda: None)
        return type.__new__(cls, name, bases, dct)