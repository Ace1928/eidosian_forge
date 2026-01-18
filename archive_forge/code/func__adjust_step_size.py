import numpy as np
import math
import inspect
import scipy.optimize
from scipy._lib._util import check_random_state
def _adjust_step_size(self):
    old_stepsize = self.takestep.stepsize
    accept_rate = float(self.naccept) / self.nstep
    if accept_rate > self.target_accept_rate:
        self.takestep.stepsize /= self.factor
    else:
        self.takestep.stepsize *= self.factor
    if self.verbose:
        print('adaptive stepsize: acceptance rate {:f} target {:f} new stepsize {:g} old stepsize {:g}'.format(accept_rate, self.target_accept_rate, self.takestep.stepsize, old_stepsize))