from django.apps.registry import apps as global_apps
from django.db import migrations, router
from .exceptions import InvalidMigrationPlan
from .loader import MigrationLoader
from .recorder import MigrationRecorder
from .state import ProjectState
def _migrate_all_forwards(self, state, plan, full_plan, fake, fake_initial):
    """
        Take a list of 2-tuples of the form (migration instance, False) and
        apply them in the order they occur in the full_plan.
        """
    migrations_to_run = {m[0] for m in plan}
    for migration, _ in full_plan:
        if not migrations_to_run:
            break
        if migration in migrations_to_run:
            if 'apps' not in state.__dict__:
                if self.progress_callback:
                    self.progress_callback('render_start')
                state.apps
                if self.progress_callback:
                    self.progress_callback('render_success')
            state = self.apply_migration(state, migration, fake=fake, fake_initial=fake_initial)
            migrations_to_run.remove(migration)
    return state