from typing import Any, Dict, List, Set
from ..language import (
class DependencyCollector(Visitor):
    dependencies: List[str]

    def __init__(self) -> None:
        super().__init__()
        self.dependencies = []
        self.add_dependency = self.dependencies.append

    def enter_fragment_spread(self, node: FragmentSpreadNode, *_args: Any) -> None:
        self.add_dependency(node.name.value)