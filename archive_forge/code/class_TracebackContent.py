import codecs
import functools
import json
import os
import traceback
from testtools.compat import _b
from testtools.content_type import ContentType, JSON, UTF8_TEXT
class TracebackContent(Content):
    """Content object for tracebacks.

    This adapts an exc_info tuple to the 'Content' interface.
    'text/x-traceback;language=python' is used for the mime type, in order to
    provide room for other languages to format their tracebacks differently.
    """

    def __init__(self, err, test, capture_locals=False):
        """Create a TracebackContent for ``err``.

        :param err: An exc_info error tuple.
        :param test: A test object used to obtain failureException.
        :param capture_locals: If true, show locals in the traceback.
        """
        if err is None:
            raise ValueError('err may not be None')
        exctype, value, tb = err
        if StackLinesContent.HIDE_INTERNAL_STACK:
            while tb and '__unittest' in tb.tb_frame.f_globals:
                tb = tb.tb_next
        limit = None
        if False and StackLinesContent.HIDE_INTERNAL_STACK and test.failureException and isinstance(value, test.failureException):
            limit = 0
            while tb and (not self._is_relevant_tb_level(tb)):
                limit += 1
                tb = tb.tb_next
        stack_lines = list(traceback.TracebackException(exctype, value, tb, limit=limit, capture_locals=capture_locals).format())
        content_type = ContentType('text', 'x-traceback', {'language': 'python', 'charset': 'utf8'})
        super().__init__(content_type, lambda: [x.encode('utf8') for x in stack_lines])