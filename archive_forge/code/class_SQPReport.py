from __future__ import annotations
class SQPReport(ReportBase):
    COLUMN_NAMES = ['niter', 'f evals', 'CG iter', 'obj func', 'tr radius', 'opt', 'c viol', 'penalty', 'CG stop']
    COLUMN_WIDTHS = [7, 7, 7, 13, 10, 10, 10, 10, 7]
    ITERATION_FORMATS = ['^7', '^7', '^7', '^+13.4e', '^10.2e', '^10.2e', '^10.2e', '^10.2e', '^7']