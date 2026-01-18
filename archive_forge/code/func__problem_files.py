import os
from pyomo.opt.base import ProblemFormat, convert_problem, guess_format
def _problem_files(self):
    if self.datfile is None:
        return [self.modfile]
    else:
        return [self.modfile, self.datfile]