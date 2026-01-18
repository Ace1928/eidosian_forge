import sys
import threading
import unittest
import gi
from gi.repository import GObject, Gtk    # noqa: E402
from testtools import StreamToExtendedDecorator  # noqa: E402
from subunit import (PROGRESS_POP, PROGRESS_PUSH, PROGRESS_SET,  # noqa: E402
from subunit.progress_model import ProgressModel  # noqa: E402
class GTKTestResult(unittest.TestResult):

    def __init__(self):
        super(GTKTestResult, self).__init__()
        self.window = None
        self.run_label = None
        self.ok_label = None
        self.not_ok_label = None
        self.total_tests = None
        self.window = Gtk.Window(Gtk.WindowType.TOPLEVEL)
        self.window.set_resizable(True)
        self.window.connect('destroy', Gtk.main_quit)
        self.window.set_title('Tests...')
        self.window.set_border_width(0)
        vbox = Gtk.VBox(False, 5)
        vbox.set_border_width(10)
        self.window.add(vbox)
        vbox.show()
        align = Gtk.Alignment.new(0.5, 0.5, 0, 0)
        vbox.pack_start(align, False, False, 5)
        align.show()
        self.pbar = Gtk.ProgressBar()
        align.add(self.pbar)
        self.pbar.set_text('Running')
        self.pbar.show()
        self.progress_model = ProgressModel()
        separator = Gtk.HSeparator()
        vbox.pack_start(separator, False, False, 0)
        separator.show()
        table = Gtk.Table(2, 3, False)
        vbox.pack_start(table, False, True, 0)
        table.show()
        label = Gtk.Label(label='Run:')
        table.attach(label, 0, 1, 1, 2, Gtk.AttachOptions.EXPAND | Gtk.AttachOptions.FILL, Gtk.AttachOptions.EXPAND | Gtk.AttachOptions.FILL, 5, 5)
        label.show()
        self.run_label = Gtk.Label(label='N/A')
        table.attach(self.run_label, 1, 2, 1, 2, Gtk.AttachOptions.EXPAND | Gtk.AttachOptions.FILL, Gtk.AttachOptions.EXPAND | Gtk.AttachOptions.FILL, 5, 5)
        self.run_label.show()
        label = Gtk.Label(label='OK:')
        table.attach(label, 0, 1, 2, 3, Gtk.AttachOptions.EXPAND | Gtk.AttachOptions.FILL, Gtk.AttachOptions.EXPAND | Gtk.AttachOptions.FILL, 5, 5)
        label.show()
        self.ok_label = Gtk.Label(label='N/A')
        table.attach(self.ok_label, 1, 2, 2, 3, Gtk.AttachOptions.EXPAND | Gtk.AttachOptions.FILL, Gtk.AttachOptions.EXPAND | Gtk.AttachOptions.FILL, 5, 5)
        self.ok_label.show()
        label = Gtk.Label(label='Not OK:')
        table.attach(label, 0, 1, 3, 4, Gtk.AttachOptions.EXPAND | Gtk.AttachOptions.FILL, Gtk.AttachOptions.EXPAND | Gtk.AttachOptions.FILL, 5, 5)
        label.show()
        self.not_ok_label = Gtk.Label(label='N/A')
        table.attach(self.not_ok_label, 1, 2, 3, 4, Gtk.AttachOptions.EXPAND | Gtk.AttachOptions.FILL, Gtk.AttachOptions.EXPAND | Gtk.AttachOptions.FILL, 5, 5)
        self.not_ok_label.show()
        self.window.show()
        self.window.set_keep_above(True)
        self.window.present()

    def stopTest(self, test):
        super(GTKTestResult, self).stopTest(test)
        GObject.idle_add(self._stopTest)

    def _stopTest(self):
        self.progress_model.advance()
        if self.progress_model.width() == 0:
            self.pbar.pulse()
        else:
            pos = self.progress_model.pos()
            width = self.progress_model.width()
            percentage = pos / float(width)
            self.pbar.set_fraction(percentage)

    def stopTestRun(self):
        try:
            super(GTKTestResult, self).stopTestRun()
        except AttributeError:
            pass
        GObject.idle_add(self.pbar.set_text, 'Finished')

    def addError(self, test, err):
        super(GTKTestResult, self).addError(test, err)
        GObject.idle_add(self.update_counts)

    def addFailure(self, test, err):
        super(GTKTestResult, self).addFailure(test, err)
        GObject.idle_add(self.update_counts)

    def addSuccess(self, test):
        super(GTKTestResult, self).addSuccess(test)
        GObject.idle_add(self.update_counts)

    def addSkip(self, test, reason):
        super(GTKTestResult, self).addSkip(test, reason)
        GObject.idle_add(self.update_counts)

    def addExpectedFailure(self, test, err):
        super(GTKTestResult, self).addExpectedFailure(test, err)
        GObject.idle_add(self.update_counts)

    def addUnexpectedSuccess(self, test):
        super(GTKTestResult, self).addUnexpectedSuccess(test)
        GObject.idle_add(self.update_counts)

    def progress(self, offset, whence):
        if whence == PROGRESS_PUSH:
            self.progress_model.push()
        elif whence == PROGRESS_POP:
            self.progress_model.pop()
        elif whence == PROGRESS_SET:
            self.total_tests = offset
            self.progress_model.set_width(offset)
        else:
            self.total_tests += offset
            self.progress_model.adjust_width(offset)

    def time(self, a_datetime):
        pass

    def update_counts(self):
        self.run_label.set_text(str(self.testsRun))
        bad = len(self.failures + self.errors)
        self.ok_label.set_text(str(self.testsRun - bad))
        self.not_ok_label.set_text(str(bad))