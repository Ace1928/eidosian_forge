import os
import re
import subprocess
import sys
import unittest
from io import BytesIO
from io import UnsupportedOperation as _UnsupportedOperation
import iso8601
from testtools import ExtendedToOriginalDecorator, content, content_type
from testtools.compat import _b, _u
from testtools.content import TracebackContent
from testtools import CopyStreamResult, testresult
from subunit import chunked, details
from subunit.v2 import ByteStreamToStreamResult, StreamResultToBytes
class TestProtocolClient(testresult.TestResult):
    """A TestResult which generates a subunit stream for a test run.

    # Get a TestSuite or TestCase to run
    suite = make_suite()
    # Create a stream (any object with a 'write' method). This should accept
    # bytes not strings: subunit is a byte orientated protocol.
    stream = file('tests.log', 'wb')
    # Create a subunit result object which will output to the stream
    result = subunit.TestProtocolClient(stream)
    # Optionally, to get timing data for performance analysis, wrap the
    # serialiser with a timing decorator
    result = subunit.test_results.AutoTimingTestResultDecorator(result)
    # Run the test suite reporting to the subunit result object
    suite.run(result)
    # Close the stream.
    stream.close()
    """

    def __init__(self, stream):
        testresult.TestResult.__init__(self)
        stream = make_stream_binary(stream)
        self._stream = stream
        self._progress_fmt = _b('progress: ')
        self._bytes_eol = _b('\n')
        self._progress_plus = _b('+')
        self._progress_push = _b('push')
        self._progress_pop = _b('pop')
        self._empty_bytes = _b('')
        self._start_simple = _b(' [\n')
        self._end_simple = _b(']\n')

    def addError(self, test, error=None, details=None):
        """Report an error in test test.

        Only one of error and details should be provided: conceptually there
        are two separate methods:
            addError(self, test, error)
            addError(self, test, details)

        :param error: Standard unittest positional argument form - an
            exc_info tuple.
        :param details: New Testing-in-python drafted API; a dict from string
            to subunit.Content objects.
        """
        self._addOutcome('error', test, error=error, details=details)
        if self.failfast:
            self.stop()

    def addExpectedFailure(self, test, error=None, details=None):
        """Report an expected failure in test test.

        Only one of error and details should be provided: conceptually there
        are two separate methods:
            addError(self, test, error)
            addError(self, test, details)

        :param error: Standard unittest positional argument form - an
            exc_info tuple.
        :param details: New Testing-in-python drafted API; a dict from string
            to subunit.Content objects.
        """
        self._addOutcome('xfail', test, error=error, details=details)

    def addFailure(self, test, error=None, details=None):
        """Report a failure in test test.

        Only one of error and details should be provided: conceptually there
        are two separate methods:
            addFailure(self, test, error)
            addFailure(self, test, details)

        :param error: Standard unittest positional argument form - an
            exc_info tuple.
        :param details: New Testing-in-python drafted API; a dict from string
            to subunit.Content objects.
        """
        self._addOutcome('failure', test, error=error, details=details)
        if self.failfast:
            self.stop()

    def _addOutcome(self, outcome, test, error=None, details=None, error_permitted=True):
        """Report a failure in test test.

        Only one of error and details should be provided: conceptually there
        are two separate methods:
            addOutcome(self, test, error)
            addOutcome(self, test, details)

        :param outcome: A string describing the outcome - used as the
            event name in the subunit stream.
        :param error: Standard unittest positional argument form - an
            exc_info tuple.
        :param details: New Testing-in-python drafted API; a dict from string
            to subunit.Content objects.
        :param error_permitted: If True then one and only one of error or
            details must be supplied. If False then error must not be supplied
            and details is still optional.  """
        self._stream.write(_b('%s: ' % outcome) + self._test_id(test))
        if error_permitted:
            if error is None and details is None:
                raise ValueError
        elif error is not None:
            raise ValueError
        if error is not None:
            self._stream.write(self._start_simple)
            tb_content = TracebackContent(error, test)
            for bytes in tb_content.iter_bytes():
                self._stream.write(bytes)
        elif details is not None:
            self._write_details(details)
        else:
            self._stream.write(_b('\n'))
        if details is not None or error is not None:
            self._stream.write(self._end_simple)

    def addSkip(self, test, reason=None, details=None):
        """Report a skipped test."""
        if reason is None:
            self._addOutcome('skip', test, error=None, details=details)
        else:
            self._stream.write(_b('skip: %s [\n' % test.id()))
            self._stream.write(_b('%s\n' % reason))
            self._stream.write(self._end_simple)

    def addSuccess(self, test, details=None):
        """Report a success in a test."""
        self._addOutcome('successful', test, details=details, error_permitted=False)

    def addUnexpectedSuccess(self, test, details=None):
        """Report an unexpected success in test test.

        Details can optionally be provided: conceptually there
        are two separate methods:
            addError(self, test)
            addError(self, test, details)

        :param details: New Testing-in-python drafted API; a dict from string
            to subunit.Content objects.
        """
        self._addOutcome('uxsuccess', test, details=details, error_permitted=False)
        if self.failfast:
            self.stop()

    def _test_id(self, test):
        result = test.id()
        if type(result) is not bytes:
            result = result.encode('utf8')
        return result

    def startTest(self, test):
        """Mark a test as starting its test run."""
        super(TestProtocolClient, self).startTest(test)
        self._stream.write(_b('test: ') + self._test_id(test) + _b('\n'))
        self._stream.flush()

    def stopTest(self, test):
        super(TestProtocolClient, self).stopTest(test)
        self._stream.flush()

    def progress(self, offset, whence):
        """Provide indication about the progress/length of the test run.

        :param offset: Information about the number of tests remaining. If
            whence is PROGRESS_CUR, then offset increases/decreases the
            remaining test count. If whence is PROGRESS_SET, then offset
            specifies exactly the remaining test count.
        :param whence: One of PROGRESS_CUR, PROGRESS_SET, PROGRESS_PUSH,
            PROGRESS_POP.
        """
        if whence == PROGRESS_CUR and offset > -1:
            prefix = self._progress_plus
            offset = _b(str(offset))
        elif whence == PROGRESS_PUSH:
            prefix = self._empty_bytes
            offset = self._progress_push
        elif whence == PROGRESS_POP:
            prefix = self._empty_bytes
            offset = self._progress_pop
        else:
            prefix = self._empty_bytes
            offset = _b(str(offset))
        self._stream.write(self._progress_fmt + prefix + offset + self._bytes_eol)

    def tags(self, new_tags, gone_tags):
        """Inform the client about tags added/removed from the stream."""
        if not new_tags and (not gone_tags):
            return
        tags = set([tag.encode('utf8') for tag in new_tags])
        tags.update([_b('-') + tag.encode('utf8') for tag in gone_tags])
        tag_line = _b('tags: ') + _b(' ').join(tags) + _b('\n')
        self._stream.write(tag_line)

    def time(self, a_datetime):
        """Inform the client of the time.

        ":param datetime: A datetime.datetime object.
        """
        time = a_datetime.astimezone(iso8601.UTC)
        self._stream.write(_b('time: %04d-%02d-%02d %02d:%02d:%02d.%06dZ\n' % (time.year, time.month, time.day, time.hour, time.minute, time.second, time.microsecond)))

    def _write_details(self, details):
        """Output details to the stream.

        :param details: An extended details dict for a test outcome.
        """
        self._stream.write(_b(' [ multipart\n'))
        for name, content in sorted(details.items()):
            self._stream.write(_b('Content-Type: %s/%s' % (content.content_type.type, content.content_type.subtype)))
            parameters = content.content_type.parameters
            if parameters:
                self._stream.write(_b(';'))
                param_strs = []
                for param, value in sorted(parameters.items()):
                    param_strs.append('%s=%s' % (param, value))
                self._stream.write(_b(','.join(param_strs)))
            self._stream.write(_b('\n%s\n' % name))
            encoder = chunked.Encoder(self._stream)
            list(map(encoder.write, content.iter_bytes()))
            encoder.close()

    def done(self):
        """Obey the testtools result.done() interface."""