import torch._dynamo.test_case
import unittest.mock
import os
import contextlib
import torch._logging
import torch._logging._internal
from torch._dynamo.utils import LazyString
import logging
def _handler_watcher(self, record_list):
    exit_stack = contextlib.ExitStack()

    def emit_post_hook(record):
        nonlocal record_list
        record_list.append(record)
    for log_qname in torch._logging._internal.log_registry.get_log_qnames():
        logger = logging.getLogger(log_qname)
        num_handlers = len(logger.handlers)
        self.assertLessEqual(num_handlers, 2, 'All pt2 loggers should only have at most two handlers (debug artifacts and messages above debug level).')
        self.assertGreater(num_handlers, 0, 'All pt2 loggers should have more than zero handlers')
        for handler in logger.handlers:
            old_emit = handler.emit

            def new_emit(record):
                old_emit(record)
                emit_post_hook(record)
            exit_stack.enter_context(unittest.mock.patch.object(handler, 'emit', new_emit))
    return exit_stack