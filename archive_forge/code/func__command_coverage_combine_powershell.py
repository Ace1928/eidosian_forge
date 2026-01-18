from __future__ import annotations
import collections.abc as c
import os
import json
import typing as t
from ...target import (
from ...io import (
from ...util import (
from ...util_common import (
from ...executor import (
from ...data import (
from ...host_configs import (
from ...provisioning import (
from . import (
def _command_coverage_combine_powershell(args: CoverageCombineConfig) -> list[str]:
    """Combine PowerShell coverage files and return a list of the output files."""
    coverage_files = get_powershell_coverage_files()

    def _default_stub_value(source_paths: list[str]) -> dict[str, dict[int, int]]:
        cmd = ['pwsh', os.path.join(ANSIBLE_TEST_TOOLS_ROOT, 'coverage_stub.ps1')]
        cmd.extend(source_paths)
        stubs = json.loads(raw_command(cmd, capture=True)[0])
        return dict(((d['Path'], dict(((line, 0) for line in d['Lines']))) for d in stubs))
    counter = 0
    sources = _get_coverage_targets(args, walk_powershell_targets)
    groups = _build_stub_groups(args, sources, _default_stub_value)
    collection_search_re, collection_sub_re = get_collection_path_regexes()
    for coverage_file in coverage_files:
        counter += 1
        display.info('[%4d/%4d] %s' % (counter, len(coverage_files), coverage_file), verbosity=2)
        group = get_coverage_group(args, coverage_file)
        if group is None:
            display.warning('Unexpected name for coverage file: %s' % coverage_file)
            continue
        for filename, hits in enumerate_powershell_lines(coverage_file, collection_search_re, collection_sub_re):
            if args.export:
                filename = os.path.relpath(filename)
            if group not in groups:
                groups[group] = {}
            coverage_data = groups[group]
            if filename not in coverage_data:
                coverage_data[filename] = {}
            file_coverage = coverage_data[filename]
            for line_no, hit_count in hits.items():
                file_coverage[line_no] = file_coverage.get(line_no, 0) + hit_count
    output_files = []
    path_checker = PathChecker(args)
    for group in sorted(groups):
        coverage_data = dict(((filename, data) for filename, data in groups[group].items() if path_checker.check_path(filename)))
        if args.all:
            missing_sources = [source for source, _source_line_count in sources if source not in coverage_data]
            coverage_data.update(_default_stub_value(missing_sources))
        if not args.explain:
            if args.export:
                output_file = os.path.join(args.export, group + '=coverage.combined')
                write_json_file(output_file, coverage_data, formatted=False)
                output_files.append(output_file)
                continue
            output_file = COVERAGE_OUTPUT_FILE_NAME + group + '-powershell'
            write_json_test_results(ResultType.COVERAGE, output_file, coverage_data, formatted=False)
            output_files.append(os.path.join(ResultType.COVERAGE.path, output_file))
    path_checker.report()
    return sorted(output_files)