from __future__ import annotations
class BasicReport(ReportBase):
    COLUMN_NAMES = ['niter', 'f evals', 'CG iter', 'obj func', 'tr radius', 'opt', 'c viol']
    COLUMN_WIDTHS = [7, 7, 7, 13, 10, 10, 10]
    ITERATION_FORMATS = ['^7', '^7', '^7', '^+13.4e', '^10.2e', '^10.2e', '^10.2e']