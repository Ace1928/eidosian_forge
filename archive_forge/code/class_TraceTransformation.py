import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
class TraceTransformation(object):
    """Print events as they pass through the transform."""

    def __init__(self, prefix='', fileobj=None):
        """Trace constructor.

        :param prefix: text to prefix each traced line with.
        :param fileobj: the writable file-like object to write to
        """
        self.prefix = prefix
        self.fileobj = fileobj or sys.stdout

    def __call__(self, stream):
        """Apply the transform filter to the marked stream.

        :param stream: the marked event stream to filter
        """
        for event in stream:
            self.fileobj.write('%s%s\n' % (self.prefix, event))
            yield event