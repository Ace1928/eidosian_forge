import ast
import os
import re
from hacking import core
from os_win.utils.winapi import libs as w_lib
import_translation_for_log_or_exception = re.compile(
class CheckForTransAdd(BaseASTChecker):
    """Checks for the use of concatenation on a translated string.

    Translations should not be concatenated with other strings, but
    should instead include the string being added to the translated
    string to give the translators the most information.
    """
    name = 'check_for_trans_add'
    version = '1.0'
    CHECK_DESC = 'N326 Translated messages cannot be concatenated.  String should be included in translated message.'
    TRANS_FUNC = ['_', '_LI', '_LW', '_LE', '_LC']

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Add):
            if self._check_call_names(node.left, self.TRANS_FUNC):
                self.add_error(node.left)
            elif self._check_call_names(node.right, self.TRANS_FUNC):
                self.add_error(node.right)
        super(CheckForTransAdd, self).generic_visit(node)