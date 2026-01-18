import sys
import collections
import copy
import importlib
import types
import warnings
import numbers
from itertools import zip_longest
from abc import ABC, abstractmethod
from typing import Dict
from Bio.Align import _pairwisealigner  # type: ignore
from Bio.Align import _codonaligner  # type: ignore
from Bio.Align import substitution_matrices
from Bio.Data import CodonTable
from Bio.Seq import Seq, MutableSeq, reverse_complement, UndefinedSequenceError
from Bio.Seq import translate
from Bio.SeqRecord import SeqRecord, _RestrictedDict
def _format_pretty(self):
    """Return default string representation (PRIVATE).

        Helper for self.format().
        """
    n = len(self.sequences)
    if n == 2:
        write_pattern = True
    else:
        write_pattern = False
    steps = np.diff(self.coordinates, 1)
    aligned = sum(steps != 0, 0) > 1
    name_width = 10
    names = []
    seqs = []
    indices = np.zeros(self.coordinates.shape, int)
    for i, (seq, positions, row) in enumerate(zip(self.sequences, self.coordinates, indices)):
        try:
            name = seq.id
            if name is None:
                raise AttributeError
        except AttributeError:
            if n == 2:
                if i == 0:
                    name = 'target'
                else:
                    name = 'query'
            else:
                name = ''
        else:
            name = name[:name_width - 1]
        name = name.ljust(name_width)
        names.append(name)
        try:
            seq = seq.seq
        except AttributeError:
            pass
        start = min(positions)
        end = max(positions)
        seq = seq[start:end]
        aligned_steps = steps[i, aligned]
        if len(aligned_steps) == 0:
            aligned_steps = steps[i]
        if sum(aligned_steps > 0) >= sum(aligned_steps < 0):
            start = min(positions)
            row[:] = positions - start
        else:
            steps[i, :] = -steps[i, :]
            seq = reverse_complement(seq)
            end = max(positions)
            row[:] = end - positions
        if isinstance(seq, str):
            if not seq.isascii():
                return self._format_unicode()
        elif isinstance(seq, (Seq, MutableSeq)):
            try:
                seq = bytes(seq)
            except UndefinedSequenceError:
                s = bytearray(b'?' * (end - start))
                for start, end in seq.defined_ranges:
                    s[start:end] = bytes(seq[start:end])
                seq = s
            seq = seq.decode()
        else:
            return self._format_generalized()
        seqs.append(seq)
    minstep = steps.min(0)
    maxstep = steps.max(0)
    steps = np.where(-minstep > maxstep, minstep, maxstep)
    for i, row in enumerate(indices):
        row_steps = np.diff(row)
        row_aligned = (row_steps > 0) & aligned
        row_steps = row_steps[row_aligned]
        aligned_steps = steps[row_aligned]
        if (row_steps == aligned_steps).all():
            pass
        elif (3 * row_steps == aligned_steps).all():
            row[:] *= 3
            seqs[i] = '  '.join(seqs[i]) + '  '
            write_pattern = False
        else:
            raise ValueError('Inconsistent coordinates')
    prefix_width = 10
    position_width = 10
    line_width = 80
    lines = []
    steps = indices[:, 1:] - indices[:, :-1]
    minstep = steps.min(0)
    maxstep = steps.max(0)
    steps = np.where(-minstep > maxstep, minstep, maxstep)
    for name, seq, positions, row in zip(names, seqs, self.coordinates, indices):
        start = positions[0]
        column = line_width
        start_index = row[0]
        for step, end, end_index in zip(steps, positions[1:], row[1:]):
            if step < 0:
                if prefix_width + position_width < column:
                    position_text = str(start)
                    offset = position_width - len(position_text) - 1
                    if offset < 0:
                        lines[-1] += ' ..' + position_text[-offset + 3:]
                    else:
                        lines[-1] += ' ' + position_text
                column = line_width
                start = end
                start_index = end_index
                continue
            elif end_index == start_index:
                s = '-' * step
            else:
                s = seq[start_index:end_index]
            while column + len(s) >= line_width:
                rest = line_width - column
                if rest > 0:
                    lines[-1] += s[:rest]
                    s = s[rest:]
                    if start != end:
                        if end_index - start_index == abs(end - start):
                            step = rest
                        else:
                            step = -(rest // -3)
                        if start < end:
                            start += step
                        else:
                            start -= step
                    start_index += rest
                line = name
                position_text = str(start)
                offset = position_width - len(position_text) - 1
                if offset < 0:
                    line += ' ..' + position_text[-offset + 3:]
                else:
                    line += ' ' * offset + position_text
                line += ' '
                lines.append(line)
                column = name_width + position_width
            lines[-1] += s
            if start_index != end_index:
                start_index = end_index
                start = end
            column += len(s)
    if write_pattern is True:
        dash = '-'
        position = 0
        m = len(lines) // 2
        lines1 = lines[:m]
        lines2 = lines[m:]
        pattern_lines = []
        for line1, line2 in zip(lines1, lines2):
            aligned_seq1 = line1[name_width + position_width:]
            aligned_seq2 = line2[name_width + position_width:]
            pattern = ''
            for c1, c2 in zip(aligned_seq1, aligned_seq2):
                if c1 == c2:
                    if c1 == ' ':
                        break
                    c = '|'
                elif c1 == dash or c2 == dash:
                    c = '-'
                else:
                    c = '.'
                pattern += c
            pattern_line = '          %9d %s' % (position, pattern)
            pattern_lines.append(pattern_line)
            position += len(pattern)
        final_position_width = len(str(max(max(self.coordinates[:, -1]), position)))
        if column + final_position_width <= line_width:
            if prefix_width + position_width < column:
                fmt = f' %{final_position_width}d'
                lines1[-1] += fmt % self.coordinates[0, -1]
                lines2[-1] += fmt % self.coordinates[1, -1]
                pattern_lines[-1] += fmt % position
        else:
            name1, name2 = names
            fmt = '%s%9d'
            line = name1 + format(self.coordinates[0, -1], '9d')
            lines1.append(line)
            line = fmt % ('          ', position)
            pattern_lines.append(line)
            line = fmt % (name2, self.coordinates[1, -1])
            lines2.append(line)
            lines.append('')
        return '\n'.join((f'{line1}\n{pattern_line}\n{line2}\n' for line1, line2, pattern_line in zip(lines1, lines2, pattern_lines)))
    else:
        m = len(lines) // n
        final_position_width = len(str(max(self.coordinates[:, -1])))
        if column + final_position_width < line_width:
            if prefix_width + position_width < column:
                fmt = f' %{final_position_width}d'
                for i in range(n):
                    lines[m - 1 + i * m] += fmt % self.coordinates[i, -1]
            blocks = ['\n'.join(lines[j::m]) + '\n' for j in range(m)]
        else:
            blocks = ['\n'.join(lines[j::m]) + '\n' for j in range(m)]
            lines = []
            fmt = '%s%9d'
            for i in range(n):
                line = names[i] + format(self.coordinates[i, -1], '9d')
                lines.append(line)
            block = '\n'.join(lines) + '\n'
            blocks.append(block)
        return '\n'.join(blocks)