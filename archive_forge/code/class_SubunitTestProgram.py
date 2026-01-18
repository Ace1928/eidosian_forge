import io
import os
import sys
from testtools import ExtendedToStreamDecorator
from testtools.run import (BUFFEROUTPUT, CATCHBREAK, FAILFAST, USAGE_AS_MAIN,
from subunit import StreamResultToBytes
from subunit.test_results import AutoTimingTestResultDecorator
class SubunitTestProgram(TestProgram):
    USAGE = USAGE_AS_MAIN

    def usageExit(self, msg=None):
        if msg:
            print(msg)
        usage = {'progName': self.progName, 'catchbreak': '', 'failfast': '', 'buffer': ''}
        if self.failfast is not False:
            usage['failfast'] = FAILFAST
        if self.catchbreak is not False:
            usage['catchbreak'] = CATCHBREAK
        if self.buffer is not False:
            usage['buffer'] = BUFFEROUTPUT
        usage_text = self.USAGE % usage
        usage_lines = usage_text.split('\n')
        usage_lines.insert(2, 'Run a test suite with a subunit reporter.')
        usage_lines.insert(3, '')
        print('\n'.join(usage_lines))
        sys.exit(2)