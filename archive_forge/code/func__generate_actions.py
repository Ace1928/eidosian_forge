from __future__ import annotations
from typing import Any
from typing import Dict
from typing import Optional
from typing import Set
from typing import TYPE_CHECKING
from . import attributes
from . import exc as orm_exc
from . import util as orm_util
from .. import event
from .. import util
from ..util import topological
def _generate_actions(self):
    """Generate the full, unsorted collection of PostSortRecs as
        well as dependency pairs for this UOWTransaction.

        """
    while True:
        ret = False
        for action in list(self.presort_actions.values()):
            if action.execute(self):
                ret = True
        if not ret:
            break
    self.cycles = cycles = topological.find_cycles(self.dependencies, list(self.postsort_actions.values()))
    if cycles:
        convert = {rec: set(rec.per_state_flush_actions(self)) for rec in cycles}
        for edge in list(self.dependencies):
            if None in edge or edge[0].disabled or edge[1].disabled or cycles.issuperset(edge):
                self.dependencies.remove(edge)
            elif edge[0] in cycles:
                self.dependencies.remove(edge)
                for dep in convert[edge[0]]:
                    self.dependencies.add((dep, edge[1]))
            elif edge[1] in cycles:
                self.dependencies.remove(edge)
                for dep in convert[edge[1]]:
                    self.dependencies.add((edge[0], dep))
    return {a for a in self.postsort_actions.values() if not a.disabled}.difference(cycles)