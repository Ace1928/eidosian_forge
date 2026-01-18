import os
import psutil
from oslo_reports.models import process as pm
class ProcessReportGenerator(object):
    """A Process Data Generator

    This generator returns a
    :class:`oslo_reports.models.process.ProcessModel`
    based on the current process (which will also include
    all subprocesses, recursively) using the :class:`psutil.Process` class`.
    """

    def __call__(self):
        return pm.ProcessModel(psutil.Process(os.getpid()))