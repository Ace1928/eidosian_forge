from __future__ import annotations
import abc
import glob
import hashlib
import json
import os
import pathlib
import re
import collections
import collections.abc as c
import typing as t
from ...constants import (
from ...encoding import (
from ...io import (
from ...util import (
from ...util_common import (
from ...ansible_util import (
from ...target import (
from ...executor import (
from ...python_requirements import (
from ...config import (
from ...test import (
from ...data import (
from ...content_config import (
from ...host_configs import (
from ...host_profiles import (
from ...provisioning import (
from ...pypi_proxy import (
from ...venv import (
class SanityCodeSmellTest(SanitySingleVersion):
    """Sanity test script."""

    def __init__(self, path) -> None:
        name = os.path.splitext(os.path.basename(path))[0]
        config_path = os.path.splitext(path)[0] + '.json'
        super().__init__(name=name)
        self.path = path
        self.config_path = config_path if os.path.exists(config_path) else None
        self.config = None
        if self.config_path:
            self.config = read_json_file(self.config_path)
        if self.config:
            self.enabled = not self.config.get('disabled')
            self.output: t.Optional[str] = self.config.get('output')
            self.extensions: list[str] = self.config.get('extensions')
            self.prefixes: list[str] = self.config.get('prefixes')
            self.files: list[str] = self.config.get('files')
            self.text: t.Optional[bool] = self.config.get('text')
            self.ignore_self: bool = self.config.get('ignore_self')
            self.minimum_python_version: t.Optional[str] = self.config.get('minimum_python_version')
            self.maximum_python_version: t.Optional[str] = self.config.get('maximum_python_version')
            self.__all_targets: bool = self.config.get('all_targets')
            self.__no_targets: bool = self.config.get('no_targets')
            self.__include_directories: bool = self.config.get('include_directories')
            self.__include_symlinks: bool = self.config.get('include_symlinks')
            self.__py2_compat: bool = self.config.get('py2_compat', False)
            self.__error_code: str | None = self.config.get('error_code', None)
        else:
            self.output = None
            self.extensions = []
            self.prefixes = []
            self.files = []
            self.text = None
            self.ignore_self = False
            self.minimum_python_version = None
            self.maximum_python_version = None
            self.__all_targets = False
            self.__no_targets = True
            self.__include_directories = False
            self.__include_symlinks = False
            self.__py2_compat = False
            self.__error_code = None
        if self.no_targets:
            mutually_exclusive = ('extensions', 'prefixes', 'files', 'text', 'ignore_self', 'all_targets', 'include_directories', 'include_symlinks')
            problems = sorted((name for name in mutually_exclusive if getattr(self, name)))
            if problems:
                raise ApplicationError('Sanity test "%s" option "no_targets" is mutually exclusive with options: %s' % (self.name, ', '.join(problems)))

    @property
    def error_code(self) -> t.Optional[str]:
        """Error code for ansible-test matching the format used by the underlying test program, or None if the program does not use error codes."""
        return self.__error_code

    @property
    def all_targets(self) -> bool:
        """True if test targets will not be filtered using includes, excludes, requires or changes. Mutually exclusive with no_targets."""
        return self.__all_targets

    @property
    def no_targets(self) -> bool:
        """True if the test does not use test targets. Mutually exclusive with all_targets."""
        return self.__no_targets

    @property
    def include_directories(self) -> bool:
        """True if the test targets should include directories."""
        return self.__include_directories

    @property
    def include_symlinks(self) -> bool:
        """True if the test targets should include symlinks."""
        return self.__include_symlinks

    @property
    def py2_compat(self) -> bool:
        """True if the test only applies to code that runs on Python 2.x."""
        return self.__py2_compat

    @property
    def supported_python_versions(self) -> t.Optional[tuple[str, ...]]:
        """A tuple of supported Python versions or None if the test does not depend on specific Python versions."""
        versions = super().supported_python_versions
        if self.minimum_python_version:
            versions = tuple((version for version in versions if str_to_version(version) >= str_to_version(self.minimum_python_version)))
        if self.maximum_python_version:
            versions = tuple((version for version in versions if str_to_version(version) <= str_to_version(self.maximum_python_version)))
        return versions

    def filter_targets(self, targets: list[TestTarget]) -> list[TestTarget]:
        """Return the given list of test targets, filtered to include only those relevant for the test."""
        if self.no_targets:
            return []
        if self.text is not None:
            if self.text:
                targets = [target for target in targets if not is_binary_file(target.path)]
            else:
                targets = [target for target in targets if is_binary_file(target.path)]
        if self.extensions:
            targets = [target for target in targets if os.path.splitext(target.path)[1] in self.extensions or (is_subdir(target.path, 'bin') and '.py' in self.extensions)]
        if self.prefixes:
            targets = [target for target in targets if any((target.path.startswith(pre) for pre in self.prefixes))]
        if self.files:
            targets = [target for target in targets if os.path.basename(target.path) in self.files]
        if self.ignore_self and data_context().content.is_ansible:
            relative_self_path = os.path.relpath(self.path, data_context().content.root)
            targets = [target for target in targets if target.path != relative_self_path]
        return targets

    def test(self, args: SanityConfig, targets: SanityTargets, python: PythonConfig) -> TestResult:
        """Run the sanity test and return the result."""
        cmd = [python.path, self.path]
        env = ansible_environment(args, color=False)
        env.update(PYTHONUTF8='1')
        pattern = None
        data = None
        settings = self.load_processor(args)
        paths = [target.path for target in targets.include]
        if self.config:
            if self.output == 'path-line-column-message':
                pattern = '^(?P<path>[^:]*):(?P<line>[0-9]+):(?P<column>[0-9]+): (?P<message>.*)$'
            elif self.output == 'path-message':
                pattern = '^(?P<path>[^:]*): (?P<message>.*)$'
            elif self.output == 'path-line-column-code-message':
                pattern = '^(?P<path>[^:]*):(?P<line>[0-9]+):(?P<column>[0-9]+): (?P<code>[^:]*): (?P<message>.*)$'
            else:
                raise ApplicationError('Unsupported output type: %s' % self.output)
        if not self.no_targets:
            data = '\n'.join(paths)
            if data:
                display.info(data, verbosity=4)
        try:
            stdout, stderr = intercept_python(args, python, cmd, data=data, env=env, capture=True)
            status = 0
        except SubprocessError as ex:
            stdout = ex.stdout
            stderr = ex.stderr
            status = ex.status
        if args.explain:
            return SanitySuccess(self.name)
        if stdout and (not stderr):
            if pattern:
                matches = parse_to_list_of_dict(pattern, stdout)
                messages = [SanityMessage(message=m['message'], path=m['path'], line=int(m.get('line', 0)), column=int(m.get('column', 0)), code=m.get('code')) for m in matches]
                messages = settings.process_errors(messages, paths)
                if not messages:
                    return SanitySuccess(self.name)
                return SanityFailure(self.name, messages=messages)
        if stderr or status:
            summary = '%s' % SubprocessError(cmd=cmd, status=status, stderr=stderr, stdout=stdout)
            return SanityFailure(self.name, summary=summary)
        messages = settings.process_errors([], paths)
        if messages:
            return SanityFailure(self.name, messages=messages)
        return SanitySuccess(self.name)

    def load_processor(self, args: SanityConfig) -> SanityIgnoreProcessor:
        """Load the ignore processor for this sanity test."""
        return SanityIgnoreProcessor(args, self, None)