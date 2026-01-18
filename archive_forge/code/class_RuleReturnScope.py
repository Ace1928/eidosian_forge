from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
import sys
from antlr3 import runtime_version, runtime_version_str
from antlr3.compat import set, frozenset, reversed
from antlr3.constants import DEFAULT_CHANNEL, HIDDEN_CHANNEL, EOF, \
from antlr3.exceptions import RecognitionException, MismatchedTokenException, \
from antlr3.tokens import CommonToken, EOF_TOKEN, SKIP_TOKEN
import six
from six import unichr
class RuleReturnScope(object):
    """
    Rules can return start/stop info as well as possible trees and templates.
    """

    def getStart(self):
        """Return the start token or tree."""
        return None

    def getStop(self):
        """Return the stop token or tree."""
        return None

    def getTree(self):
        """Has a value potentially if output=AST."""
        return None

    def getTemplate(self):
        """Has a value potentially if output=template."""
        return None