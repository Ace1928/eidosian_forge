from unittest import TestCase
from ipykernel.tests import utils
from nbformat.converter import convert
from nbformat.reader import reads
import re
import json
from copy import copy
import unittest
def _test_notebook_cell(self, cell, i, kernel, test):
    if hasattr(cell, 'source'):
        code = cell.source
    else:
        code = cell.input
    iopub = kernel.iopub_channel
    kernel.execute(code)
    outputs = []
    msg = None
    no_error = True
    first_error = -1
    error_msg = ''
    while self.should_continue(msg):
        try:
            msg = iopub.get_msg(block=True, timeout=1)
        except Empty:
            continue
        if msg['msg_type'] not in self.IGNORE_TYPES:
            if msg['msg_type'] == 'error':
                error_msg = '  ' + msg['content']['ename'] + '\n  ' + msg['content']['evalue']
                no_error = False
                if first_error == -1:
                    first_error = i
            i = len(outputs)
            expected = i < len(cell.outputs) and cell.outputs[i] or []
            o = self.transform_message(msg, expected)
            outputs.append(o)
    if test == 'check_error':
        self.assertTrue(no_error, 'Executing cell %d resulted in an error:\n%s' % (first_error, error_msg))
    else:
        scrub = lambda x: self.dump_canonical(list(self.scrub_outputs(x)))
        scrubbed = scrub(outputs)
        expected = scrub(cell.outputs)