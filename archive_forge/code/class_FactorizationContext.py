from pyomo.contrib.pynumero.interfaces.utils import (
import numpy as np
import logging
import time
from pyomo.contrib.pynumero.linalg.base import LinearSolverStatus
from pyomo.common.timing import HierarchicalTimer
import enum
class FactorizationContext(object):

    def __init__(self, logger):
        self.logger = logger

    def __enter__(self):
        self.logger.debug('Factorizing KKT')
        self.log_header()
        return self

    def __exit__(self, et, ev, tb):
        self.logger.debug('Finished factorizing KKT')

    def log_header(self):
        self.logger.debug('{_iter:<10}{reg_iter:<10}{num_realloc:<10}{reg_coef:<10}{neg_eig:<10}{status:<10}'.format(_iter='Iter', reg_iter='reg_iter', num_realloc='# realloc', reg_coef='reg_coef', neg_eig='neg_eig', status='status'))

    def log_info(self, _iter, reg_iter, num_realloc, coef, neg_eig, status):
        self.logger.debug('{_iter:<10}{reg_iter:<10}{num_realloc:<10}{reg_coef:<10.2e}{neg_eig:<10}{status:<10}'.format(_iter=_iter, reg_iter=reg_iter, num_realloc=num_realloc, reg_coef=coef, neg_eig=str(neg_eig), status=status.name))