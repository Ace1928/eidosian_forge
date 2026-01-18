from django.apps.registry import apps as global_apps
from django.db import migrations, router
from .exceptions import InvalidMigrationPlan
from .loader import MigrationLoader
from .recorder import MigrationRecorder
from .state import ProjectState
def _create_project_state(self, with_applied_migrations=False):
    """
        Create a project state including all the applications without
        migrations and applied migrations if with_applied_migrations=True.
        """
    state = ProjectState(real_apps=self.loader.unmigrated_apps)
    if with_applied_migrations:
        full_plan = self.migration_plan(self.loader.graph.leaf_nodes(), clean_start=True)
        applied_migrations = {self.loader.graph.nodes[key] for key in self.loader.applied_migrations if key in self.loader.graph.nodes}
        for migration, _ in full_plan:
            if migration in applied_migrations:
                migration.mutate_state(state, preserve=False)
    return state