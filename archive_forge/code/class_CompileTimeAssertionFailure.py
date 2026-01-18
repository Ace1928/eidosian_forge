import ast
from typing import Optional, Union
class CompileTimeAssertionFailure(CompilationError):
    """Specific exception for failed tests in `static_assert` invocations"""
    pass