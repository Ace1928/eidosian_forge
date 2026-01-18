import ast
from hacking import core
import re
class CheckForMutableDefaultArgs(BaseASTChecker):
    """Check for the use of mutable objects as function/method defaults.

    We are only checking for list and dict literals at this time. This means
    that a developer could specify an instance of their own and cause a bug.
    The fix for this is probably more work than it's worth because it will
    get caught during code review.

    """
    name = 'check_for_mutable_default_args'
    version = '1.0'
    CHECK_DESC = 'K001 Using mutable as a function/method default'
    MUTABLES = (ast.List, ast.ListComp, ast.Dict, ast.DictComp, ast.Set, ast.SetComp, ast.Call)

    def visit_FunctionDef(self, node):
        for arg in node.args.defaults:
            if isinstance(arg, self.MUTABLES):
                self.add_error(arg)
        super(CheckForMutableDefaultArgs, self).generic_visit(node)