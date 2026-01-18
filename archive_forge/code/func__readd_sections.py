import inspect
import logging
import os
import signal
import stat
import sys
import threading
import time
import traceback
from oslo_utils import timeutils
from oslo_reports.generators import conf as cgen
from oslo_reports.generators import process as prgen
from oslo_reports.generators import threading as tgen
from oslo_reports.generators import version as pgen
from oslo_reports import report
def _readd_sections(self):
    del self.sections[self.start_section_index:]
    self.add_section('Package', pgen.PackageReportGenerator(self.version_obj))
    self.add_section('Threads', tgen.ThreadReportGenerator(self.traceback))
    if greenlet:
        self.add_section('Green Threads', tgen.GreenThreadReportGenerator())
    self.add_section('Processes', prgen.ProcessReportGenerator())
    self.add_section('Configuration', cgen.ConfigReportGenerator())
    try:
        for section_title, generator in self.persistent_sections:
            self.add_section(section_title, generator)
    except AttributeError:
        pass