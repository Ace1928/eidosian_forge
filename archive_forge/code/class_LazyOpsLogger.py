import os
import sys
import signal
import threading
import logging
from dataclasses import dataclass
from functools import partialmethod
from typing import Optional, Any, Dict, List
from .mp_utils import _CPU_CORES, _MAX_THREADS, _MAX_PROCS
class LazyOpsLogger:

    def __init__(self, config):
        self.config = config
        self.logger = self.setup_logging()

    def setup_logging(self):
        logger = logging.getLogger(self.config['name'])
        logger.setLevel(_logging_levels[self.config.get('log_level', 'info')])
        console_log_output = sys.stdout if _notebook or _colab else sys.stderr
        console_handler = logging.StreamHandler(console_log_output)
        console_handler.setLevel(self.config['console_log_level'].upper())
        console_formatter = LogFormatter(fmt=self.config['log_line_template'], color=self.config['console_log_color'])
        console_handler.setFormatter(console_formatter)
        if self.config.get('clear_handlers', False) and logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(console_handler)
        if self.config.get('quiet_loggers'):
            to_quiet = self.config['quiet_loggers']
            if isinstance(to_quiet, str):
                to_quiet = [to_quiet]
            for clr in to_quiet:
                clr_logger = logging.getLogger(clr)
                clr_logger.setLevel(logging.ERROR)
        logger.propagate = self.config.get('propagate', False)
        return logger

    def get_logger(self, module=None):
        if module:
            return self.logger.getChild(module)
        return self.logger

    def debug(self, msg, *args, **kwargs):
        return self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        return self.logger.info(msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        return self.logger.warn(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        return self.logger.error(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        return self.logger.exception(msg, *args, **kwargs)

    def __call__(self, msg, *args, **kwargs):
        return self.logger.info(msg, *args, **kwargs)