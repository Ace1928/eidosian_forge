import sys
import re
import os
from configparser import RawConfigParser
class LibraryInfo:
    """
    Object containing build information about a library.

    Parameters
    ----------
    name : str
        The library name.
    description : str
        Description of the library.
    version : str
        Version string.
    sections : dict
        The sections of the configuration file for the library. The keys are
        the section headers, the values the text under each header.
    vars : class instance
        A `VariableSet` instance, which contains ``(name, value)`` pairs for
        variables defined in the configuration file for the library.
    requires : sequence, optional
        The required libraries for the library to be installed.

    Notes
    -----
    All input parameters (except "sections" which is a method) are available as
    attributes of the same name.

    """

    def __init__(self, name, description, version, sections, vars, requires=None):
        self.name = name
        self.description = description
        if requires:
            self.requires = requires
        else:
            self.requires = []
        self.version = version
        self._sections = sections
        self.vars = vars

    def sections(self):
        """
        Return the section headers of the config file.

        Parameters
        ----------
        None

        Returns
        -------
        keys : list of str
            The list of section headers.

        """
        return list(self._sections.keys())

    def cflags(self, section='default'):
        val = self.vars.interpolate(self._sections[section]['cflags'])
        return _escape_backslash(val)

    def libs(self, section='default'):
        val = self.vars.interpolate(self._sections[section]['libs'])
        return _escape_backslash(val)

    def __str__(self):
        m = ['Name: %s' % self.name, 'Description: %s' % self.description]
        if self.requires:
            m.append('Requires:')
        else:
            m.append('Requires: %s' % ','.join(self.requires))
        m.append('Version: %s' % self.version)
        return '\n'.join(m)