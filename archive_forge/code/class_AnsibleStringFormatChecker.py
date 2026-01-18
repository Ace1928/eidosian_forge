from __future__ import annotations
import astroid
from pylint.checkers import BaseChecker
from pylint.checkers import utils
class AnsibleStringFormatChecker(BaseChecker):
    """Checks string formatting operations to ensure that the format string
    is valid and the arguments match the format string.
    """
    __implements__ = (IAstroidChecker,)
    name = 'string'
    msgs = MSGS

    @check_messages(*MSGS.keys())
    def visit_call(self, node):
        """Visit a call node."""
        func = utils.safe_infer(node.func)
        if isinstance(func, astroid.BoundMethod) and isinstance(func.bound, astroid.Instance) and (func.bound.name in ('str', 'unicode', 'bytes')):
            if func.name == 'format':
                self._check_new_format(node, func)

    def _check_new_format(self, node, func):
        """ Check the new string formatting """
        if isinstance(node.func, astroid.Attribute) and (not isinstance(node.func.expr, astroid.Const)):
            return
        try:
            strnode = next(func.bound.infer())
        except astroid.InferenceError:
            return
        if not isinstance(strnode, astroid.Const):
            return
        if isinstance(strnode.value, bytes):
            self.add_message('ansible-no-format-on-bytestring', node=node)
            return