import posixpath
import re
from codecs import open
from collections import OrderedDict
from os import sep
from os.path import dirname, normpath
from editorconfig.compat import u
from editorconfig.exceptions import ParsingError
from editorconfig.fnmatch import fnmatch
class EditorConfigParser(object):
    """Parser for EditorConfig-style configuration files

    Based on RawConfigParser from ConfigParser.py in Python 2.6.
    """
    SECTCRE = re.compile('\n\n        \\s *                                # Optional whitespace\n        \\[                                  # Opening square brace\n\n        (?P<header>                         # One or more characters excluding\n            ( [^\\#;] | \\\\\\# | \\\\; ) +       # unescaped # and ; characters\n        )\n\n        \\]                                  # Closing square brace\n\n        ', re.VERBOSE)
    OPTCRE = re.compile('\n\n        \\s *                                # Optional whitespace\n        (?P<option>                         # One or more characters excluding\n            [^:=\\s]                         # : a = characters (and first\n            [^:=] *                         # must not be whitespace)\n        )\n        \\s *                                # Optional whitespace\n        (?P<vi>\n            [:=]                            # Single = or : character\n        )\n        \\s *                                # Optional whitespace\n        (?P<value>\n            . *                             # One or more characters\n        )\n        $\n\n        ', re.VERBOSE)

    def __init__(self, filename):
        self.filename = filename
        self.options = OrderedDict()
        self.root_file = False

    def matches_filename(self, config_filename, glob):
        """Return True if section glob matches filename"""
        config_dirname = normpath(dirname(config_filename)).replace(sep, '/')
        glob = glob.replace('\\#', '#')
        glob = glob.replace('\\;', ';')
        if '/' in glob:
            if glob.find('/') == 0:
                glob = glob[1:]
            glob = posixpath.join(config_dirname, glob)
        else:
            glob = posixpath.join('**/', glob)
        return fnmatch(self.filename, glob)

    def read(self, filename):
        """Read and parse single EditorConfig file"""
        try:
            fp = open(filename, encoding='utf-8')
        except IOError:
            return
        self._read(fp, filename)
        fp.close()

    def _read(self, fp, fpname):
        """Parse a sectioned setup file.

        The sections in setup file contains a title line at the top,
        indicated by a name in square brackets (`[]'), plus key/value
        options lines, indicated by `name: value' format lines.
        Continuations are represented by an embedded newline then
        leading whitespace.  Blank lines, lines beginning with a '#',
        and just about everything else are ignored.
        """
        in_section = False
        matching_section = False
        optname = None
        lineno = 0
        e = None
        while True:
            line = fp.readline()
            if not line:
                break
            if lineno == 0 and line.startswith(u('\ufeff')):
                line = line[1:]
            lineno = lineno + 1
            if line.strip() == '' or line[0] in '#;':
                continue
            else:
                mo = self.SECTCRE.match(line)
                if mo:
                    sectname = mo.group('header')
                    if len(sectname) > MAX_SECTION_LENGTH:
                        continue
                    in_section = True
                    matching_section = self.matches_filename(fpname, sectname)
                    optname = None
                else:
                    mo = self.OPTCRE.match(line)
                    if mo:
                        optname, vi, optval = mo.group('option', 'vi', 'value')
                        if ';' in optval or '#' in optval:
                            m = re.search('(.*?) [;#]', optval)
                            if m:
                                optval = m.group(1)
                        optval = optval.strip()
                        if optval == '""':
                            optval = ''
                        optname = self.optionxform(optname.rstrip())
                        if len(optname) > MAX_PROPERTY_LENGTH or len(optval) > MAX_VALUE_LENGTH:
                            continue
                        if not in_section and optname == 'root':
                            self.root_file = optval.lower() == 'true'
                        if matching_section:
                            self.options[optname] = optval
                    else:
                        if not e:
                            e = ParsingError(fpname)
                        e.append(lineno, repr(line))
        if e:
            raise e

    def optionxform(self, optionstr):
        return optionstr.lower()