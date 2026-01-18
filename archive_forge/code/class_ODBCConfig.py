import os.path
import re
import sys
import shutil
from decimal import Decimal
from pyomo.common.dependencies import attempt_import
from pyomo.dataportal import TableData
from pyomo.dataportal.factory import DataManagerFactory
class ODBCConfig:
    """
    Encapsulates an ODBC configuration file, usually odbc.ini or
    .odbc.ini, as specified by IBM. ODBC config data can be loaded
    either from a file or a string containing the relevant formatted
    data. Calling load() after initialization will update existing
    information in the config object with the new information.
    """

    def __init__(self, filename=None, data=None):
        """
        Create a new ODBC config instance, loading data from
        the given file and/or data string. Once initialized, the
        new config will contain the data represented in both
        arguments, if any. Data specified as a string argument
        will override that in the file.
        """
        self.ODBC_DS_KEY = 'ODBC Data Sources'
        self.ODBC_INFO_KEY = 'ODBC'
        self.file = filename
        self.sources = {}
        self.source_specs = {}
        self.odbc_info = {}
        self.load(self.file, data)

    def load(self, filename=None, data=None):
        """
        Load data from the given file and/or data string. If
        both are given, data contained in the string will override
        that in the file. If this config object already contains
        data, the new information loaded will update the old,
        replacing where keys are the same.
        """
        sections = {}
        if filename is not None:
            with open(filename, 'r') as fileHandle:
                fileData = fileHandle.read()
                sections.update(self._get_sections(fileData))
        if data is not None:
            sections.update(self._get_sections(data))
        if self.ODBC_DS_KEY in sections:
            self.sources.update(sections[self.ODBC_DS_KEY])
            del sections[self.ODBC_DS_KEY]
        if self.ODBC_INFO_KEY in sections:
            self.odbc_info.update(sections[self.ODBC_INFO_KEY])
            del sections[self.ODBC_INFO_KEY]
        self.source_specs.update(sections)

    def __str__(self):
        return '<ODBC config: {0} sources, {1} source specs>'.format(len(self.sources), len(self.source_specs))

    def __eq__(self, other):
        if isinstance(other, ODBCConfig):
            return self.sources == other.sources and self.source_specs == other.source_specs and (self.odbc_info == other.odbc_info)
        return False

    def odbc_repr(self):
        """
        Get the full, odbc.ini-style representation of this
        ODBC configuration.
        """
        str = '[{0}]\n'.format(self.ODBC_DS_KEY)
        for name in self.sources:
            str += '{0} = {1}\n'.format(name, self.sources[name])
        for name in self.source_specs:
            str += '\n[{0}]\n'.format(name)
            for key in self.source_specs[name]:
                str += '{0} = {1}\n'.format(key, self.source_specs[name][key])
        if len(self.odbc_info) > 0:
            str += '\n[{0}]\n'.format(self.ODBC_INFO_KEY)
            for key in self.odbc_info:
                str += '{0} = {1}\n'.format(key, self.odbc_info[key])
        return str

    def write(self, filename):
        """
        Write the current ODBC configuration to the given file.
        Depends on the odbc_repr() function for a string
        representation of the stored ODBC config information.
        """
        with open(filename, 'w') as f:
            f.write(self.odbc_repr())

    def add_source(self, name, driver):
        """
        Add an ODBC data source to the configuration. A data
        source consists of a unique source name and a driver
        string, which specifies how the source will be loaded.
        If a source name is not unique, it will replace the
        existing source of the same name. A source is required
        in order to have a source specification.
        """
        if name is None or driver is None or len(name) == 0 or (len(driver) == 0):
            raise ODBCError('A source must specify both a name and a driver string')
        if name == self.ODBC_DS_KEY or name == self.ODBC_INFO_KEY:
            raise ODBCError("A source cannot use the reserved name '{0}'".format(name))
        self.sources[str(name)] = str(driver)

    def del_source(self, name):
        """
        Remove an ODBC data source from the configuration. If
        any source specifications are based on this source, they
        will be removed as well.
        """
        if name in self.sources:
            if name in self.source_specs:
                del self.source_specs[name]
            del self.sources[name]

    def add_source_spec(self, name, spec):
        """
        Add an ODBC data source specification to the configuration.
        A source specification consists of a unique name and
        a key-value mapping (i.e. dictionary) of options. In order
        to add a source specification, a data source with a matching
        name must exist in the configuration.
        """
        if name is None or spec is None or len(name) == 0:
            raise ODBCError('A source spec must specify both a name and a spec dictionary')
        if name not in self.sources:
            raise ODBCError('A source spec must have a corresponding source; call .add_source() first')
        self.source_specs[name] = dict(spec)

    def del_source_spec(self, name):
        """
        Remove an ODBC data source specification from the
        configuration.
        """
        if name in self.source_specs:
            del self.source_specs[name]

    def set_odbc_info(self, key, value):
        """
        Set an option for the ODBC handling specified in the
        configuration. An option consists of a key-value pair.
        Specifying an existing key will update the current value.
        """
        if key is None or value is None or len(key) == 0 or (len(value) == 0):
            raise ODBCError('An ODBC info pair must specify both a key and a value')
        self.odbc_info[str(key)] = str(value)

    def _get_sections(self, data):
        """
        Parse a string for ODBC sections. The parsing algorithm proceeds
        roughly as follows:

        1. Split the string on newline ('\\n') characters.
        2. Remove lines consisting purely of whitespace.
        3. Iterate over lines, storing all key-value pair lines in a dictionary.
        4. When reaching a new section header (denoted by '[str]'), store the old
           key-value pairs under the old section name. Continue from step 3.
        5. On reaching end of data, store the last section and return a mapping
           from section names to dictionaries of key-value pairs in those sections.
        """
        sections = {}
        sectionKey = None
        sectionContents = {}
        emptyLine = re.compile('^[ \t\r]*$')
        lines = data.split('\n')
        for line in lines:
            if emptyLine.match(line):
                pass
            elif len(line) < 2:
                raise ODBCError('Malformed line in ODBC config (no meaningful data): ' + line)
            elif line[0] == '[' and line[-1] == ']':
                sections[sectionKey] = sectionContents
                sectionKey = line[1:-1]
                sectionContents = {}
            else:
                if '=' not in line:
                    raise ODBCError('Malformed line in ODBC config (no key-value mapping): ' + line)
                key, value = line.split('=', 1)
                if '=' in value:
                    raise ODBCError("Malformed line in ODBC config (too many '='): " + line)
                sectionContents[key.strip()] = value.strip()
        sections[sectionKey] = sectionContents
        if None in sections:
            del sections[None]
        return sections