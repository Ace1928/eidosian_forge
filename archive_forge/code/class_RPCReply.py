from threading import Event, Lock
from uuid import uuid4
from ncclient.xml_ import *
from ncclient.logging_ import SessionLoggerAdapter
from ncclient.transport import SessionListener
from ncclient.operations import util
from ncclient.operations.errors import OperationError, TimeoutExpiredError, MissingCapabilityError
import logging
class RPCReply(object):
    """Represents an *rpc-reply*. Only concerns itself with whether the operation was successful.

    *raw*: the raw unparsed reply

    *huge_tree*: parse XML with very deep trees and very long text content

    .. note::
        If the reply has not yet been parsed there is an implicit, one-time parsing overhead to
        accessing some of the attributes defined by this class.
    """
    ERROR_CLS = RPCError
    'Subclasses can specify a different error class, but it should be a subclass of `RPCError`.'

    def __init__(self, raw, huge_tree=False, parsing_error_transform=None):
        self._raw = raw
        self._parsing_error_transform = parsing_error_transform
        self._parsed = False
        self._root = None
        self._errors = []
        self._huge_tree = huge_tree

    def __repr__(self):
        return self._raw

    def parse(self):
        """Parses the *rpc-reply*."""
        if self._parsed:
            return
        root = self._root = to_ele(self._raw, huge_tree=self._huge_tree)
        ok = root.find(qualify('ok'))
        if ok is None:
            error = root.find('.//' + qualify('rpc-error'))
            if error is not None:
                for err in root.getiterator(error.tag):
                    self._errors.append(self.ERROR_CLS(err))
        try:
            self._parsing_hook(root)
        except Exception as e:
            if self._parsing_error_transform is None:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                six.reraise(exc_type, exc_value, exc_traceback)
            self._parsing_error_transform(root)
            self._parsing_hook(root)
        self._parsed = True

    def _parsing_hook(self, root):
        """No-op by default. Gets passed the *root* element for the reply."""
        pass

    def set_parsing_error_transform(self, transform_function):
        self._parsing_error_transform = transform_function

    @property
    def xml(self):
        """*rpc-reply* element as returned."""
        return self._raw

    @property
    def ok(self):
        """Boolean value indicating if there were no errors."""
        self.parse()
        return not self.errors

    @property
    def error(self):
        """Returns the first :class:`RPCError` and `None` if there were no errors."""
        self.parse()
        if self._errors:
            return self._errors[0]
        else:
            return None

    @property
    def errors(self):
        """List of `RPCError` objects. Will be empty if there were no *rpc-error* elements in reply."""
        self.parse()
        return self._errors