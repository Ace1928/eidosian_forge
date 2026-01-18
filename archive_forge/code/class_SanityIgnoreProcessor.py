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
class SanityIgnoreProcessor:
    """Processor for sanity test ignores for a single run of one sanity test."""

    def __init__(self, args: SanityConfig, test: SanityTest, python_version: t.Optional[str]) -> None:
        name = test.name
        code = test.error_code
        if python_version:
            full_name = '%s-%s' % (name, python_version)
        else:
            full_name = name
        self.args = args
        self.test = test
        self.code = code
        self.parser = SanityIgnoreParser.load(args)
        self.ignore_entries = self.parser.ignores.get(full_name, {})
        self.skip_entries = self.parser.skips.get(full_name, {})
        self.used_line_numbers: set[int] = set()

    def filter_skipped_targets(self, targets: list[TestTarget]) -> list[TestTarget]:
        """Return the given targets, with any skipped paths filtered out."""
        return sorted((target for target in targets if target.path not in self.skip_entries))

    def process_errors(self, errors: list[SanityMessage], paths: list[str]) -> list[SanityMessage]:
        """Return the given errors filtered for ignores and with any settings related errors included."""
        errors = self.filter_messages(errors)
        errors.extend(self.get_errors(paths))
        errors = sorted(set(errors))
        return errors

    def filter_messages(self, messages: list[SanityMessage]) -> list[SanityMessage]:
        """Return a filtered list of the given messages using the entries that have been loaded."""
        filtered = []
        for message in messages:
            if message.code in self.test.optional_error_codes and (not self.args.enable_optional_errors):
                continue
            path_entry = self.ignore_entries.get(message.path)
            if path_entry:
                code = message.code if self.code else SanityIgnoreParser.NO_CODE
                line_no = path_entry.get(code)
                if line_no:
                    self.used_line_numbers.add(line_no)
                    continue
            filtered.append(message)
        return filtered

    def get_errors(self, paths: list[str]) -> list[SanityMessage]:
        """Return error messages related to issues with the file."""
        messages: list[SanityMessage] = []
        unused: list[tuple[int, str, str]] = []
        if self.test.no_targets or self.test.all_targets:
            targets = SanityTargets.get_targets()
            test_targets = SanityTargets.filter_and_inject_targets(self.test, targets)
            paths = [target.path for target in test_targets]
        for path in paths:
            path_entry = self.ignore_entries.get(path)
            if not path_entry:
                continue
            unused.extend(((line_no, path, code) for code, line_no in path_entry.items() if line_no not in self.used_line_numbers))
        messages.extend((SanityMessage(code=self.code, message="Ignoring '%s' on '%s' is unnecessary" % (code, path) if self.code else "Ignoring '%s' is unnecessary" % path, path=self.parser.relative_path, line=line, column=1, confidence=calculate_best_confidence(((self.parser.path, line), (path, 0)), self.args.metadata) if self.args.metadata.changes else None) for line, path, code in unused))
        return messages