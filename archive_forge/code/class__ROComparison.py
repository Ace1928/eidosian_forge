class _ROComparison:

    class Item:
        prefix = '  '

        def __init__(self, item):
            self.item = item

        def __str__(self):
            return '{}{}'.format(self.prefix, self.item)

    class Deleted(Item):
        prefix = '- '

    class Inserted(Item):
        prefix = '+ '
    Empty = str

    class ReplacedBy:
        prefix = '- '
        suffix = ''

        def __init__(self, chunk, total_count):
            self.chunk = chunk
            self.total_count = total_count

        def __iter__(self):
            lines = [self.prefix + str(item) + self.suffix for item in self.chunk]
            while len(lines) < self.total_count:
                lines.append('')
            return iter(lines)

    class Replacing(ReplacedBy):
        prefix = '+ '
        suffix = ''
    _c3_report = None
    _legacy_report = None

    def __init__(self, c3, c3_ro, legacy_ro):
        self.c3 = c3
        self.c3_ro = c3_ro
        self.legacy_ro = legacy_ro

    def __move(self, from_, to_, chunk, operation):
        for x in chunk:
            to_.append(operation(x))
            from_.append(self.Empty())

    def _generate_report(self):
        if self._c3_report is None:
            import difflib
            matcher = difflib.SequenceMatcher(None, self.legacy_ro, self.c3_ro)
            self._c3_report = c3_report = []
            self._legacy_report = legacy_report = []
            for opcode, leg1, leg2, c31, c32 in matcher.get_opcodes():
                c3_chunk = self.c3_ro[c31:c32]
                legacy_chunk = self.legacy_ro[leg1:leg2]
                if opcode == 'equal':
                    c3_report.extend((self.Item(x) for x in c3_chunk))
                    legacy_report.extend((self.Item(x) for x in legacy_chunk))
                if opcode == 'delete':
                    assert not c3_chunk
                    self.__move(c3_report, legacy_report, legacy_chunk, self.Deleted)
                if opcode == 'insert':
                    assert not legacy_chunk
                    self.__move(legacy_report, c3_report, c3_chunk, self.Inserted)
                if opcode == 'replace':
                    chunk_size = max(len(c3_chunk), len(legacy_chunk))
                    c3_report.extend(self.Replacing(c3_chunk, chunk_size))
                    legacy_report.extend(self.ReplacedBy(legacy_chunk, chunk_size))
        return (self._c3_report, self._legacy_report)

    @property
    def _inconsistent_label(self):
        inconsistent = []
        if self.c3.direct_inconsistency:
            inconsistent.append('direct')
        if self.c3.bases_had_inconsistency:
            inconsistent.append('bases')
        return '+'.join(inconsistent) if inconsistent else 'no'

    def __str__(self):
        c3_report, legacy_report = self._generate_report()
        assert len(c3_report) == len(legacy_report)
        left_lines = [str(x) for x in legacy_report]
        right_lines = [str(x) for x in c3_report]
        assert len(left_lines) == len(right_lines)
        padding = ' ' * 2
        max_left = max((len(x) for x in left_lines))
        max_right = max((len(x) for x in right_lines))
        left_title = 'Legacy RO (len={})'.format(len(self.legacy_ro))
        right_title = 'C3 RO (len={}; inconsistent={})'.format(len(self.c3_ro), self._inconsistent_label)
        lines = [padding + left_title.ljust(max_left) + padding + right_title.ljust(max_right), padding + '=' * (max_left + len(padding) + max_right)]
        lines += [padding + left.ljust(max_left) + padding + right for left, right in zip(left_lines, right_lines)]
        return '\n'.join(lines)