from __future__ import annotations
class ReportBase:
    COLUMN_NAMES: list[str] = NotImplemented
    COLUMN_WIDTHS: list[int] = NotImplemented
    ITERATION_FORMATS: list[str] = NotImplemented

    @classmethod
    def print_header(cls):
        fmt = '|' + '|'.join([f'{{:^{x}}}' for x in cls.COLUMN_WIDTHS]) + '|'
        separators = ['-' * x for x in cls.COLUMN_WIDTHS]
        print(fmt.format(*cls.COLUMN_NAMES))
        print(fmt.format(*separators))

    @classmethod
    def print_iteration(cls, *args):
        iteration_format = [f'{{:{x}}}' for x in cls.ITERATION_FORMATS]
        fmt = '|' + '|'.join(iteration_format) + '|'
        print(fmt.format(*args))

    @classmethod
    def print_footer(cls):
        print()