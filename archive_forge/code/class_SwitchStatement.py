import enum
from typing import Optional, List, Union, Iterable, Tuple
class SwitchStatement(Statement):
    """AST node for the stable 'switch' statement of OpenQASM 3.

    The only real difference from an AST form is that the default is required to be separate; it
    cannot be joined with other cases (even though that's meaningless, the V1 syntax permitted it).
    """

    def __init__(self, target: Expression, cases: Iterable[Tuple[Iterable[Expression], ProgramBlock]], default: Optional[ProgramBlock]=None):
        self.target = target
        self.cases = [(tuple(values), case) for values, case in cases]
        self.default = default