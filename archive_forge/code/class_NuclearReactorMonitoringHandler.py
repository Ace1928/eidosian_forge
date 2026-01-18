import logging
import os
import pathlib
import sys
import time
import pytest
class NuclearReactorMonitoringHandler(logging.Handler):
    NUCLEAR_REACTOR_STATUS = 'Nominal'

    def __init__(self):
        super().__init__(level=logging.CRITICAL)

    def emit(self, log_record):
        sys.stderr.write('Please proceed immediately to the nearest exit.\n')
        sys.stderr.flush()
        self.NUCLEAR_REACTOR_STATUS = 'Evacuated'