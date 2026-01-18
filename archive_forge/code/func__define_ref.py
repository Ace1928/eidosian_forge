import typing as t
from . import nodes
from .visitor import NodeVisitor
def _define_ref(self, name: str, load: t.Optional[t.Tuple[str, t.Optional[str]]]=None) -> str:
    ident = f'l_{self.level}_{name}'
    self.refs[name] = ident
    if load is not None:
        self.loads[ident] = load
    return ident