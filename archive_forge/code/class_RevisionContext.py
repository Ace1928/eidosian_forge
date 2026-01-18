from __future__ import annotations
import contextlib
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import inspect
from . import compare
from . import render
from .. import util
from ..operations import ops
from ..util import sqla_compat
class RevisionContext:
    """Maintains configuration and state that's specific to a revision
    file generation operation."""
    generated_revisions: List[MigrationScript]
    process_revision_directives: Optional[ProcessRevisionDirectiveFn]

    def __init__(self, config: Config, script_directory: ScriptDirectory, command_args: Dict[str, Any], process_revision_directives: Optional[ProcessRevisionDirectiveFn]=None) -> None:
        self.config = config
        self.script_directory = script_directory
        self.command_args = command_args
        self.process_revision_directives = process_revision_directives
        self.template_args = {'config': config}
        self.generated_revisions = [self._default_revision()]

    def _to_script(self, migration_script: MigrationScript) -> Optional[Script]:
        template_args: Dict[str, Any] = self.template_args.copy()
        if getattr(migration_script, '_needs_render', False):
            autogen_context = self._last_autogen_context
            autogen_context.imports = set()
            if migration_script.imports:
                autogen_context.imports.update(migration_script.imports)
            render._render_python_into_templatevars(autogen_context, migration_script, template_args)
        assert migration_script.rev_id is not None
        return self.script_directory.generate_revision(migration_script.rev_id, migration_script.message, refresh=True, head=migration_script.head, splice=migration_script.splice, branch_labels=migration_script.branch_label, version_path=migration_script.version_path, depends_on=migration_script.depends_on, **template_args)

    def run_autogenerate(self, rev: _GetRevArg, migration_context: MigrationContext) -> None:
        self._run_environment(rev, migration_context, True)

    def run_no_autogenerate(self, rev: _GetRevArg, migration_context: MigrationContext) -> None:
        self._run_environment(rev, migration_context, False)

    def _run_environment(self, rev: _GetRevArg, migration_context: MigrationContext, autogenerate: bool) -> None:
        if autogenerate:
            if self.command_args['sql']:
                raise util.CommandError('Using --sql with --autogenerate does not make any sense')
            if set(self.script_directory.get_revisions(rev)) != set(self.script_directory.get_revisions('heads')):
                raise util.CommandError('Target database is not up to date.')
        upgrade_token = migration_context.opts['upgrade_token']
        downgrade_token = migration_context.opts['downgrade_token']
        migration_script = self.generated_revisions[-1]
        if not getattr(migration_script, '_needs_render', False):
            migration_script.upgrade_ops_list[-1].upgrade_token = upgrade_token
            migration_script.downgrade_ops_list[-1].downgrade_token = downgrade_token
            migration_script._needs_render = True
        else:
            migration_script._upgrade_ops.append(ops.UpgradeOps([], upgrade_token=upgrade_token))
            migration_script._downgrade_ops.append(ops.DowngradeOps([], downgrade_token=downgrade_token))
        autogen_context = AutogenContext(migration_context, autogenerate=autogenerate)
        self._last_autogen_context: AutogenContext = autogen_context
        if autogenerate:
            compare._populate_migration_script(autogen_context, migration_script)
        if self.process_revision_directives:
            self.process_revision_directives(migration_context, rev, self.generated_revisions)
        hook = migration_context.opts['process_revision_directives']
        if hook:
            hook(migration_context, rev, self.generated_revisions)
        for migration_script in self.generated_revisions:
            migration_script._needs_render = True

    def _default_revision(self) -> MigrationScript:
        command_args: Dict[str, Any] = self.command_args
        op = ops.MigrationScript(rev_id=command_args['rev_id'] or util.rev_id(), message=command_args['message'], upgrade_ops=ops.UpgradeOps([]), downgrade_ops=ops.DowngradeOps([]), head=command_args['head'], splice=command_args['splice'], branch_label=command_args['branch_label'], version_path=command_args['version_path'], depends_on=command_args['depends_on'])
        return op

    def generate_scripts(self) -> Iterator[Optional[Script]]:
        for generated_revision in self.generated_revisions:
            yield self._to_script(generated_revision)