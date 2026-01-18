import os
import pwd
import re
import subprocess
import time
import xml.dom.minidom
import random
from ... import logging
from ...interfaces.base import CommandLine
from .base import SGELikeBatchManagerBase, logger
def _remove_old_jobs(self):
    """This is only called during initialization of the function for the purpose
        of identifying jobs that are not part of this run of nipype.  They
        are jobs that existed prior to starting a new jobs, so they are irrelevant.
        """
    self._run_qstat('QstatInitialization', True)