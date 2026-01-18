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
class TextGuruMeditation(GuruMeditation, report.TextReport):
    """A Text Guru Meditation Report

    This report is the basic human-readable Guru Meditation Report

    It contains the following sections by default
    (in addition to any registered persistent sections):

    - Package Information

    - Threads List

    - Green Threads List

    - Process List

    - Configuration Options

    :param version_obj: the version object for the current product
    :param traceback: an (optional) frame object providing the actual
                      traceback for the current thread
    """

    def __init__(self, version_obj, traceback=None):
        super(TextGuruMeditation, self).__init__(version_obj, traceback, 'Guru Meditation')