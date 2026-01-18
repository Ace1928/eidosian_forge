from functools import total_ordering
from django.db.migrations.state import ProjectState
from .exceptions import CircularDependencyError, NodeNotFoundError
def _generate_plan(self, nodes, at_end):
    plan = []
    for node in nodes:
        for migration in self.forwards_plan(node):
            if migration not in plan and (at_end or migration not in nodes):
                plan.append(migration)
    return plan