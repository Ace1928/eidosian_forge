from __future__ import annotations
import base64
import hashlib
import sys
from typing import IO, Iterable, TYPE_CHECKING
from coverage.plugin import FileReporter
from coverage.report_core import get_analysis_to_report
from coverage.results import Analysis, Numbers
from coverage.types import TMorf
class LcovReporter:
    """A reporter for writing LCOV coverage reports."""
    report_type = 'LCOV report'

    def __init__(self, coverage: Coverage) -> None:
        self.coverage = coverage
        self.total = Numbers(self.coverage.config.precision)

    def report(self, morfs: Iterable[TMorf] | None, outfile: IO[str]) -> float:
        """Renders the full lcov report.

        `morfs` is a list of modules or filenames

        outfile is the file object to write the file into.
        """
        self.coverage.get_data()
        outfile = outfile or sys.stdout
        for fr, analysis in get_analysis_to_report(self.coverage, morfs):
            self.get_lcov(fr, analysis, outfile)
        return self.total.n_statements and self.total.pc_covered

    def get_lcov(self, fr: FileReporter, analysis: Analysis, outfile: IO[str]) -> None:
        """Produces the lcov data for a single file.

        This currently supports both line and branch coverage,
        however function coverage is not supported.
        """
        self.total += analysis.numbers
        outfile.write('TN:\n')
        outfile.write(f'SF:{fr.relative_filename()}\n')
        source_lines = fr.source().splitlines()
        for covered in sorted(analysis.executed):
            if covered in analysis.excluded:
                continue
            if source_lines:
                if covered - 1 >= len(source_lines):
                    break
                line = source_lines[covered - 1]
            else:
                line = ''
            outfile.write(f'DA:{covered},1,{line_hash(line)}\n')
        for missed in sorted(analysis.missing):
            assert source_lines
            line = source_lines[missed - 1]
            outfile.write(f'DA:{missed},0,{line_hash(line)}\n')
        outfile.write(f'LF:{analysis.numbers.n_statements}\n')
        outfile.write(f'LH:{analysis.numbers.n_executed}\n')
        missing_arcs = analysis.missing_branch_arcs()
        executed_arcs = analysis.executed_branch_arcs()
        for block_number, block_line_number in enumerate(sorted(analysis.branch_stats().keys())):
            for branch_number, line_number in enumerate(sorted(missing_arcs[block_line_number])):
                line_number = max(line_number, 0)
                outfile.write(f'BRDA:{line_number},{block_number},{branch_number},-\n')
            for branch_number, line_number in enumerate(sorted(executed_arcs[block_line_number]), start=len(missing_arcs[block_line_number])):
                line_number = max(line_number, 0)
                outfile.write(f'BRDA:{line_number},{block_number},{branch_number},1\n')
        if analysis.has_arcs():
            branch_stats = analysis.branch_stats()
            brf = sum((t for t, k in branch_stats.values()))
            brh = brf - sum((t - k for t, k in branch_stats.values()))
            outfile.write(f'BRF:{brf}\n')
            outfile.write(f'BRH:{brh}\n')
        outfile.write('end_of_record\n')