from __future__ import annotations
import collections
import collections.abc as c
import enum
import os
import re
import itertools
import abc
import typing as t
from .encoding import (
from .io import (
from .util import (
from .data import (
class IntegrationTarget(CompletionTarget):
    """Integration test target."""
    non_posix = frozenset(('network', 'windows'))
    categories = frozenset(non_posix | frozenset(('posix', 'module', 'needs', 'skip')))

    def __init__(self, path: str, modules: frozenset[str], prefixes: dict[str, str]) -> None:
        super().__init__()
        self.relative_path = os.path.relpath(path, data_context().content.integration_targets_path)
        self.name = self.relative_path.replace(os.path.sep, '.')
        self.path = path
        file_paths = data_context().content.get_files(path)
        runme_path = os.path.join(path, 'runme.sh')
        if runme_path in file_paths:
            self.type = 'script'
            self.script_path = runme_path
        else:
            self.type = 'role'
            self.script_path = None
        aliases_path = os.path.join(path, 'aliases')
        if aliases_path in file_paths:
            static_aliases = tuple(read_lines_without_comments(aliases_path, remove_blank_lines=True))
        else:
            static_aliases = tuple()
        if self.name in modules:
            module_name = self.name
        elif self.name.startswith('win_') and self.name[4:] in modules:
            module_name = self.name[4:]
        else:
            module_name = None
        self.modules = tuple(sorted((a for a in static_aliases + tuple([module_name]) if a in modules)))
        groups = [self.type]
        groups += [a for a in static_aliases if a not in modules]
        groups += ['module/%s' % m for m in self.modules]
        if data_context().content.is_ansible and (self.name == 'ansible-test' or self.name.startswith('ansible-test-')):
            groups.append('ansible-test')
        if not self.modules:
            groups.append('non_module')
        if 'destructive' not in groups:
            groups.append('non_destructive')
        if 'needs/httptester' in groups:
            groups.append('cloud/httptester')
        for prefix, group in prefixes.items():
            if not self.name.startswith(f'{prefix}_'):
                continue
            if group != prefix:
                group = '%s/%s' % (group, prefix)
            groups.append(group)
        if self.name.startswith('win_'):
            groups.append('windows')
        if self.name.startswith('connection_'):
            groups.append('connection')
        if self.name.startswith('setup_') or self.name.startswith('prepare_'):
            groups.append('hidden')
        if self.type not in ('script', 'role'):
            groups.append('hidden')
        targets_relative_path = data_context().content.integration_targets_path
        self.skips = tuple((g for g in groups if g.startswith('skip/')))
        self.needs_file = tuple(sorted(set(('/'.join(g.split('/')[2:]) for g in groups if g.startswith('needs/file/') and (not g.startswith('needs/file/%s/' % targets_relative_path))))))
        networks = [g.split('/')[1] for g in groups if g.startswith('network/')]
        self.network_platform = networks[0] if networks else None
        for group in itertools.islice(groups, 0, len(groups)):
            if '/' in group:
                parts = group.split('/')
                for i in range(1, len(parts)):
                    groups.append('/'.join(parts[:i]))
        if not any((g in self.non_posix for g in groups)):
            groups.append('posix')
        force_target = any((group in self.non_posix for group in groups))
        target_type, actual_type = categorize_integration_test(self.name, list(static_aliases), force_target)
        groups.extend(['context/', f'context/{target_type.name.lower()}'])
        if target_type != actual_type:
            groups.extend(['context/', f'context/{actual_type.name.lower()}'])
        self.target_type = target_type
        self.actual_type = actual_type
        aliases = [self.name] + ['%s/' % g for g in groups] + ['%s/%s' % (g, self.name) for g in groups if g not in self.categories]
        if 'hidden/' in aliases:
            aliases = ['hidden/'] + ['hidden/%s' % a for a in aliases if not a.startswith('hidden/')]
        self.aliases = tuple(sorted(set(aliases)))
        self.retry_never = 'retry/never/' in self.aliases
        self.setup_once = tuple(sorted(set((g.split('/')[2] for g in groups if g.startswith('setup/once/')))))
        self.setup_always = tuple(sorted(set((g.split('/')[2] for g in groups if g.startswith('setup/always/')))))
        self.needs_target = tuple(sorted(set((g.split('/')[2] for g in groups if g.startswith('needs/target/')))))