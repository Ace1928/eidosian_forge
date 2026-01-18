from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
class Diagnostic(object):
    """
    A Diagnostic is a single instance of a Clang diagnostic. It includes the
    diagnostic severity, the message, the location the diagnostic occurred, as
    well as additional source ranges and associated fix-it hints.
    """
    Ignored = 0
    Note = 1
    Warning = 2
    Error = 3
    Fatal = 4
    DisplaySourceLocation = 1
    DisplayColumn = 2
    DisplaySourceRanges = 4
    DisplayOption = 8
    DisplayCategoryId = 16
    DisplayCategoryName = 32
    _FormatOptionsMask = 63

    def __init__(self, ptr):
        self.ptr = ptr

    def __del__(self):
        conf.lib.clang_disposeDiagnostic(self)

    @property
    def severity(self):
        return conf.lib.clang_getDiagnosticSeverity(self)

    @property
    def location(self):
        return conf.lib.clang_getDiagnosticLocation(self)

    @property
    def spelling(self):
        return conf.lib.clang_getDiagnosticSpelling(self)

    @property
    def ranges(self):

        class RangeIterator(object):

            def __init__(self, diag):
                self.diag = diag

            def __len__(self):
                return int(conf.lib.clang_getDiagnosticNumRanges(self.diag))

            def __getitem__(self, key):
                if key >= len(self):
                    raise IndexError
                return conf.lib.clang_getDiagnosticRange(self.diag, key)
        return RangeIterator(self)

    @property
    def fixits(self):

        class FixItIterator(object):

            def __init__(self, diag):
                self.diag = diag

            def __len__(self):
                return int(conf.lib.clang_getDiagnosticNumFixIts(self.diag))

            def __getitem__(self, key):
                range = SourceRange()
                value = conf.lib.clang_getDiagnosticFixIt(self.diag, key, byref(range))
                if len(value) == 0:
                    raise IndexError
                return FixIt(range, value)
        return FixItIterator(self)

    @property
    def children(self):

        class ChildDiagnosticsIterator(object):

            def __init__(self, diag):
                self.diag_set = conf.lib.clang_getChildDiagnostics(diag)

            def __len__(self):
                return int(conf.lib.clang_getNumDiagnosticsInSet(self.diag_set))

            def __getitem__(self, key):
                diag = conf.lib.clang_getDiagnosticInSet(self.diag_set, key)
                if not diag:
                    raise IndexError
                return Diagnostic(diag)
        return ChildDiagnosticsIterator(self)

    @property
    def category_number(self):
        """The category number for this diagnostic or 0 if unavailable."""
        return conf.lib.clang_getDiagnosticCategory(self)

    @property
    def category_name(self):
        """The string name of the category for this diagnostic."""
        return conf.lib.clang_getDiagnosticCategoryText(self)

    @property
    def option(self):
        """The command-line option that enables this diagnostic."""
        return conf.lib.clang_getDiagnosticOption(self, None)

    @property
    def disable_option(self):
        """The command-line option that disables this diagnostic."""
        disable = _CXString()
        conf.lib.clang_getDiagnosticOption(self, byref(disable))
        return _CXString.from_result(disable)

    def format(self, options=None):
        """
        Format this diagnostic for display. The options argument takes
        Diagnostic.Display* flags, which can be combined using bitwise OR. If
        the options argument is not provided, the default display options will
        be used.
        """
        if options is None:
            options = conf.lib.clang_defaultDiagnosticDisplayOptions()
        if options & ~Diagnostic._FormatOptionsMask:
            raise ValueError('Invalid format options')
        return conf.lib.clang_formatDiagnostic(self, options)

    def __repr__(self):
        return '<Diagnostic severity %r, location %r, spelling %r>' % (self.severity, self.location, self.spelling)

    def __str__(self):
        return self.format()

    def from_param(self):
        return self.ptr