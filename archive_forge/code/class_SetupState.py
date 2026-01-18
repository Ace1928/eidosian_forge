import bdb
import dataclasses
import os
import sys
from typing import Callable
from typing import cast
from typing import Dict
from typing import final
from typing import Generic
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .reports import BaseReport
from .reports import CollectErrorRepr
from .reports import CollectReport
from .reports import TestReport
from _pytest import timing
from _pytest._code.code import ExceptionChainRepr
from _pytest._code.code import ExceptionInfo
from _pytest._code.code import TerminalRepr
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.nodes import Collector
from _pytest.nodes import Directory
from _pytest.nodes import Item
from _pytest.nodes import Node
from _pytest.outcomes import Exit
from _pytest.outcomes import OutcomeException
from _pytest.outcomes import Skipped
from _pytest.outcomes import TEST_OUTCOME
class SetupState:
    """Shared state for setting up/tearing down test items or collectors
    in a session.

    Suppose we have a collection tree as follows:

    <Session session>
        <Module mod1>
            <Function item1>
        <Module mod2>
            <Function item2>

    The SetupState maintains a stack. The stack starts out empty:

        []

    During the setup phase of item1, setup(item1) is called. What it does
    is:

        push session to stack, run session.setup()
        push mod1 to stack, run mod1.setup()
        push item1 to stack, run item1.setup()

    The stack is:

        [session, mod1, item1]

    While the stack is in this shape, it is allowed to add finalizers to
    each of session, mod1, item1 using addfinalizer().

    During the teardown phase of item1, teardown_exact(item2) is called,
    where item2 is the next item to item1. What it does is:

        pop item1 from stack, run its teardowns
        pop mod1 from stack, run its teardowns

    mod1 was popped because it ended its purpose with item1. The stack is:

        [session]

    During the setup phase of item2, setup(item2) is called. What it does
    is:

        push mod2 to stack, run mod2.setup()
        push item2 to stack, run item2.setup()

    Stack:

        [session, mod2, item2]

    During the teardown phase of item2, teardown_exact(None) is called,
    because item2 is the last item. What it does is:

        pop item2 from stack, run its teardowns
        pop mod2 from stack, run its teardowns
        pop session from stack, run its teardowns

    Stack:

        []

    The end!
    """

    def __init__(self) -> None:
        self.stack: Dict[Node, Tuple[List[Callable[[], object]], Optional[Union[OutcomeException, Exception]]]] = {}

    def setup(self, item: Item) -> None:
        """Setup objects along the collector chain to the item."""
        needed_collectors = item.listchain()
        for col, (finalizers, exc) in self.stack.items():
            assert col in needed_collectors, 'previous item was not torn down properly'
            if exc:
                raise exc
        for col in needed_collectors[len(self.stack):]:
            assert col not in self.stack
            self.stack[col] = ([col.teardown], None)
            try:
                col.setup()
            except TEST_OUTCOME as exc:
                self.stack[col] = (self.stack[col][0], exc)
                raise exc

    def addfinalizer(self, finalizer: Callable[[], object], node: Node) -> None:
        """Attach a finalizer to the given node.

        The node must be currently active in the stack.
        """
        assert node and (not isinstance(node, tuple))
        assert callable(finalizer)
        assert node in self.stack, (node, self.stack)
        self.stack[node][0].append(finalizer)

    def teardown_exact(self, nextitem: Optional[Item]) -> None:
        """Teardown the current stack up until reaching nodes that nextitem
        also descends from.

        When nextitem is None (meaning we're at the last item), the entire
        stack is torn down.
        """
        needed_collectors = nextitem and nextitem.listchain() or []
        exceptions: List[BaseException] = []
        while self.stack:
            if list(self.stack.keys()) == needed_collectors[:len(self.stack)]:
                break
            node, (finalizers, _) = self.stack.popitem()
            these_exceptions = []
            while finalizers:
                fin = finalizers.pop()
                try:
                    fin()
                except TEST_OUTCOME as e:
                    these_exceptions.append(e)
            if len(these_exceptions) == 1:
                exceptions.extend(these_exceptions)
            elif these_exceptions:
                msg = f'errors while tearing down {node!r}'
                exceptions.append(BaseExceptionGroup(msg, these_exceptions[::-1]))
        if len(exceptions) == 1:
            raise exceptions[0]
        elif exceptions:
            raise BaseExceptionGroup('errors during test teardown', exceptions[::-1])
        if nextitem is None:
            assert not self.stack