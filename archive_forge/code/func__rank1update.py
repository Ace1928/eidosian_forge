import copy
from math import sqrt, log, exp
from itertools import cycle
import warnings
import numpy
from . import tools
def _rank1update(self, individual, p_succ):
    update_cov = False
    self.psucc = (1 - self.cp) * self.psucc + self.cp * p_succ
    if not hasattr(self.parent, 'fitness') or self.parent.fitness <= individual.fitness:
        self.parent = copy.deepcopy(individual)
        self.ancestors_fitness.append(copy.deepcopy(individual.fitness))
        if len(self.ancestors_fitness) > 5:
            self.ancestors_fitness.pop()
        if self.psucc < self.pthresh or numpy.allclose(self.pc, 0):
            self.pc = (1 - self.cc) * self.pc + numpy.sqrt(self.cc * (2 - self.cc)) * individual._y
            a = numpy.sqrt(1 - self.ccovp)
            w = numpy.dot(self.invA, self.pc)
            w_norm_sqrd = numpy.linalg.norm(w) ** 2
            b = numpy.sqrt(1 - self.ccovp) / w_norm_sqrd * (numpy.sqrt(1 + self.ccovp / (1 - self.ccovp) * w_norm_sqrd) - 1)
        else:
            self.pc = (1 - self.cc) * self.pc
            d = self.ccovp * (1 + self.cc * (2 - self.cc))
            a = numpy.sqrt(1 - d)
            w = numpy.dot(self.invA, self.pc)
            w_norm_sqrd = numpy.linalg.norm(w) ** 2
            b = numpy.sqrt(1 - d) * (numpy.sqrt(1 + self.ccovp * w_norm_sqrd / (1 - d)) - 1) / w_norm_sqrd
        update_cov = True
    elif len(self.ancestors_fitness) >= 5 and individual.fitness < self.ancestors_fitness[0] and (self.psucc < self.pthresh):
        w = individual._z
        w_norm_sqrd = numpy.linalg.norm(w) ** 2
        if 1 < self.ccovn * (2 * w_norm_sqrd - 1):
            ccovn = 1 / (2 * w_norm_sqrd - 1)
        else:
            ccovn = self.ccovn
        a = numpy.sqrt(1 + ccovn)
        b = numpy.sqrt(1 + ccovn) / w_norm_sqrd * (numpy.sqrt(1 - ccovn / (1 + ccovn) * w_norm_sqrd) - 1)
        update_cov = True
    if update_cov:
        self.A = self.A * a + b * numpy.outer(numpy.dot(self.A, w), w)
        self.invA = 1 / a * self.invA - b / (a ** 2 + a * b * w_norm_sqrd) * numpy.dot(self.invA, numpy.outer(w, w))
    self.sigma = self.sigma * numpy.exp(1.0 / self.d * ((self.psucc - self.ptarg) / (1.0 - self.ptarg)))