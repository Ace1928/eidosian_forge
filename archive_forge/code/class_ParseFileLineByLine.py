import sys
import os
import getopt
from pyparsing import *
class ParseFileLineByLine:
    """
    Bring data from text files into a program, optionally parsing each line
    according to specifications in a parse definition file.

    ParseFileLineByLine instances can be used like normal file objects (i.e. by
    calling readline(), readlines(), and write()), but can also be used as
    sequences of lines in for-loops.

    ParseFileLineByLine objects also handle compression transparently. i.e. it
    is possible to read lines from a compressed text file as if it were not
    compressed.  Compression is deduced from the file name suffixes '.Z'
    (compress/uncompress), '.gz' (gzip/gunzip), and '.bz2' (bzip2).

    The parse definition fi le name is developed based on the input file name.
    If the input file name is 'basename.ext', then the definition file is
    'basename_def.ext'.  If a definition file specific to the input file is not
    found, then the program searches for the file 'sparse.def' which would be
    the definition file for all files in that directory without a file specific
    definition file.

    Finally, ParseFileLineByLine objects accept file names that start with '~'
    or '~user' to indicate a home directory, as well as URLs (for reading
    only).

    Constructor:
    ParseFileLineByLine(|filename|, |mode|='"r"'), where |filename| is the name
    of the file (or a URL) and |mode| is one of '"r"' (read), '"w"' (write) or
    '"a"' (append, not supported for .Z files).
    """

    def __init__(self, filename, mode='r'):
        """Opens input file, and if available the definition file.  If the
        definition file is available __init__ will then create some pyparsing
        helper variables.  """
        if mode not in ['r', 'w', 'a']:
            raise IOError(0, 'Illegal mode: ' + repr(mode))
        if string.find(filename, ':/') > 1:
            if mode == 'w':
                raise IOError("can't write to a URL")
            import urllib.request, urllib.parse, urllib.error
            self.file = urllib.request.urlopen(filename)
        else:
            filename = os.path.expanduser(filename)
            if mode == 'r' or mode == 'a':
                if not os.path.exists(filename):
                    raise IOError(2, 'No such file or directory: ' + filename)
            filen, file_extension = os.path.splitext(filename)
            command_dict = {('.Z', 'r'): "self.file = os.popen('uncompress -c ' + filename, mode)", ('.gz', 'r'): "self.file = gzip.GzipFile(filename, 'rb')", ('.bz2', 'r'): "self.file = os.popen('bzip2 -dc ' + filename, mode)", ('.Z', 'w'): "self.file = os.popen('compress > ' + filename, mode)", ('.gz', 'w'): "self.file = gzip.GzipFile(filename, 'wb')", ('.bz2', 'w'): "self.file = os.popen('bzip2 > ' + filename, mode)", ('.Z', 'a'): "raise IOError, (0, 'Can't append to .Z files')", ('.gz', 'a'): "self.file = gzip.GzipFile(filename, 'ab')", ('.bz2', 'a'): "raise IOError, (0, 'Can't append to .bz2 files')"}
            exec(command_dict.get((file_extension, mode), 'self.file = open(filename, mode)'))
        self.grammar = None
        definition_file_one = filen + '_def' + file_extension
        definition_file_two = os.path.dirname(filen) + os.sep + 'sparse.def'
        if os.path.exists(definition_file_one):
            self.parsedef = definition_file_one
        elif os.path.exists(definition_file_two):
            self.parsedef = definition_file_two
        else:
            self.parsedef = None
            return None
        decimal_sep = '.'
        sign = oneOf('+ -')
        special_chars = string.replace('!"#$%&\'()*,./:;<=>?@[\\]^_`{|}~', decimal_sep, '')
        integer = ToInteger(Combine(Optional(sign) + Word(nums))).setName('integer')
        positive_integer = ToInteger(Combine(Optional('+') + Word(nums))).setName('integer')
        negative_integer = ToInteger(Combine('-' + Word(nums))).setName('integer')
        real = ToFloat(Combine(Optional(sign) + Word(nums) + decimal_sep + Optional(Word(nums)) + Optional(oneOf('E e') + Word(nums)))).setName('real')
        positive_real = ToFloat(Combine(Optional('+') + Word(nums) + decimal_sep + Optional(Word(nums)) + Optional(oneOf('E e') + Word(nums)))).setName('real')
        negative_real = ToFloat(Combine('-' + Word(nums) + decimal_sep + Optional(Word(nums)) + Optional(oneOf('E e') + Word(nums)))).setName('real')
        qString = (sglQuotedString | dblQuotedString).setName('qString')
        integer_junk = Optional(Suppress(Word(alphas + special_chars + decimal_sep))).setName('integer_junk')
        real_junk = Optional(Suppress(Word(alphas + special_chars))).setName('real_junk')
        qString_junk = SkipTo(qString).setName('qString_junk')
        exec(compile(open(self.parsedef).read(), self.parsedef, 'exec'))
        grammar = []
        for nam, expr in parse:
            grammar.append(eval(expr.name + '_junk'))
            grammar.append(expr.setResultsName(nam))
        self.grammar = And(grammar[1:] + [restOfLine])

    def __del__(self):
        """Delete (close) the file wrapper."""
        self.close()

    def __getitem__(self, item):
        """Used in 'for line in fp:' idiom."""
        line = self.readline()
        if not line:
            raise IndexError
        return line

    def readline(self):
        """Reads (and optionally parses) a single line."""
        line = self.file.readline()
        if self.grammar and line:
            try:
                return self.grammar.parseString(line).asDict()
            except ParseException:
                return self.readline()
        else:
            return line

    def readlines(self):
        """Returns a list of all lines (optionally parsed) in the file."""
        if self.grammar:
            tot = []
            while 1:
                line = self.file.readline()
                if not line:
                    break
                tot.append(line)
            return tot
        return self.file.readlines()

    def write(self, data):
        """Write to a file."""
        self.file.write(data)

    def writelines(self, list):
        """Write a list to a file. Each item in the list is a line in the
        file.
        """
        for line in list:
            self.file.write(line)

    def close(self):
        """Close the file."""
        self.file.close()

    def flush(self):
        """Flush in memory contents to file."""
        self.file.flush()