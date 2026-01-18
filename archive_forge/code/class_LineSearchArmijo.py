import logging
import math
import numpy as np
from ase.utils import longsum
class LineSearchArmijo:

    def __init__(self, func, c1=0.1, tol=1e-14):
        """Initialise the linesearch with set parameters and functions.

        Args:
            func: the function we are trying to minimise (energy), which should
                take an array of positions for its argument
            c1: parameter for the sufficient decrease condition in (0.0 0.5)
            tol: tolerance for evaluating equality

        """
        self.tol = tol
        self.func = func
        if not 0 < c1 < 0.5:
            logger.error('c1 outside of allowed interval (0, 0.5). Replacing with default value.')
            print('Warning: C1 outside of allowed interval. Replacing with default value.')
            c1 = 0.1
        self.c1 = c1

    def run(self, x_start, dirn, a_max=None, a_min=None, a1=None, func_start=None, func_old=None, func_prime_start=None, rigid_units=None, rotation_factors=None, maxstep=None):
        """Perform a backtracking / quadratic-interpolation linesearch
            to find an appropriate step length with Armijo condition.
        NOTE THIS LINESEARCH DOES NOT IMPOSE WOLFE CONDITIONS!

        The idea is to do backtracking via quadratic interpolation, stabilised
        by putting a lower bound on the decrease at each linesearch step.
        To ensure BFGS-behaviour, whenever "reasonable" we take 1.0 as the
        starting step.

        Since Armijo does not guarantee convergence of BFGS, the outer
        BFGS algorithm must restart when the current search direction
        ceases to be a descent direction.

        Args:
            x_start: vector containing the position to begin the linesearch
                from (ie the current location of the optimisation)
            dirn: vector pointing in the direction to search in (pk in [NW]).
                Note that this does not have to be a unit vector, but the
                function will return a value scaled with respect to dirn.
            a_max: an upper bound on the maximum step length allowed. Default is 2.0.
            a_min: a lower bound on the minimum step length allowed. Default is 1e-10.
                A RuntimeError is raised if this bound is violated 
                during the line search.
            a1: the initial guess for an acceptable step length. If no value is
                given, this will be set automatically, using quadratic
                interpolation using func_old, or "rounded" to 1.0 if the
                initial guess lies near 1.0. (specifically for LBFGS)
            func_start: the value of func at the start of the linesearch, ie
                phi(0). Passing this information avoids potentially expensive
                re-calculations
            func_prime_start: the value of func_prime at the start of the
                linesearch (this will be dotted with dirn to find phi_prime(0))
            func_old: the value of func_start at the previous step taken in
                the optimisation (this will be used to calculate the initial
                guess for the step length if it is not provided)
            rigid_units, rotationfactors : see documentation of RumPath, if it is
                unclear what these parameters are, then leave them at None
            maxstep: maximum allowed displacement in Angstrom. Default is 0.2.

        Returns:
            A tuple: (step, func_val, no_update)

            step: the final chosen step length, representing the number of
                multiples of the direction vector to move
            func_val: the value of func after taking this step, ie phi(step)
            no_update: true if the linesearch has not performed any updates of
                phi or alpha, due to errors or immediate convergence

        Raises:
            ValueError for problems with arguments
            RuntimeError for problems encountered during iteration
        """
        a1 = self.handle_args(x_start, dirn, a_max, a_min, a1, func_start, func_old, func_prime_start, maxstep)
        logger.debug('a1(auto) = ', a1)
        if abs(a1 - 1.0) <= 0.5:
            a1 = 1.0
        logger.debug('-----------NEW LINESEARCH STARTED---------')
        a_final = None
        phi_a_final = None
        num_iter = 0
        if rigid_units is None:
            logger.debug('-----using LinearPath-----')
            path = LinearPath(dirn)
        else:
            logger.debug('-----using RumPath------')
            if rotation_factors == None:
                raise RuntimeError('RumPath cannot be created since rotation_factors == None')
            path = RumPath(x_start, dirn, rigid_units, rotation_factors)
        while True:
            logger.debug('-----------NEW ITERATION OF LINESEARCH----------')
            logger.debug('Number of linesearch iterations: %d', num_iter)
            logger.debug('a1 = %e', a1)
            func_a1 = self.func(x_start + path.step(a1))
            phi_a1 = func_a1
            suff_dec = phi_a1 <= self.func_start + self.c1 * a1 * self.phi_prime_start
            logger.info('a1 = %.3f, suff_dec = %r', a1, suff_dec)
            if a1 < self.a_min:
                raise RuntimeError('a1 < a_min, giving up')
            if self.phi_prime_start > 0.0:
                raise RuntimeError('self.phi_prime_start > 0.0')
            if suff_dec:
                a_final = a1
                phi_a_final = phi_a1
                logger.debug('Linesearch returned a = %e, phi_a = %e', a_final, phi_a_final)
                logger.debug('-----------LINESEARCH COMPLETE-----------')
                return (a_final, phi_a_final, num_iter == 0)
            at = -(self.phi_prime_start * a1 / (2 * ((phi_a1 - self.func_start) / a1 - self.phi_prime_start)))
            logger.debug('quadratic_min: initial at = %e', at)
            a1 = max(at, a1 / 10.0)
            if a1 > at:
                logger.debug('at (%e) < a1/10: revert to backtracking a1/10', at)

    def handle_args(self, x_start, dirn, a_max, a_min, a1, func_start, func_old, func_prime_start, maxstep):
        """Verify passed parameters and set appropriate attributes accordingly.

        A suitable value for the initial step-length guess will be either
        verified or calculated, stored in the attribute self.a_start, and
        returned.

        Args:
            The args should be identical to those of self.run().

        Returns:
            The suitable initial step-length guess a_start

        Raises:
            ValueError for problems with arguments

        """
        self.a_max = a_max
        self.a_min = a_min
        self.x_start = x_start
        self.dirn = dirn
        self.func_old = func_old
        self.func_start = func_start
        self.func_prime_start = func_prime_start
        if a_max is None:
            a_max = 2.0
        if a_max < self.tol:
            logger.warning('a_max too small relative to tol. Reverting to default value a_max = 2.0 (twice the <ideal> step).')
            a_max = 2.0
        if self.a_min is None:
            self.a_min = 1e-10
        if func_start is None:
            logger.debug('Setting func_start')
            self.func_start = self.func(x_start)
        self.phi_prime_start = longsum(self.func_prime_start * self.dirn)
        if self.phi_prime_start >= 0:
            logger.error('Passed direction which is not downhill. Aborting...')
            raise ValueError('Direction is not downhill.')
        elif math.isinf(self.phi_prime_start):
            logger.error('Passed func_prime_start and dirn which are too big. Aborting...')
            raise ValueError('func_prime_start and dirn are too big.')
        if a1 is None:
            if func_old is not None:
                a1 = 2 * (self.func_start - self.func_old) / self.phi_prime_start
                logger.debug('Interpolated quadratic, obtained a1 = %e', a1)
        if a1 is None or a1 > a_max:
            logger.debug('a1 greater than a_max. Reverting to default value a1 = 1.0')
            a1 = 1.0
        if a1 is None or a1 < self.tol:
            logger.debug('a1 is None or a1 < self.tol. Reverting to default value a1 = 1.0')
            a1 = 1.0
        if a1 is None or a1 < self.a_min:
            logger.debug('a1 is None or a1 < a_min. Reverting to default value a1 = 1.0')
            a1 = 1.0
        if maxstep is None:
            maxstep = 0.2
        logger.debug('maxstep = %e', maxstep)
        r = np.reshape(dirn, (-1, 3))
        steplengths = ((a1 * r) ** 2).sum(1) ** 0.5
        maxsteplength = np.max(steplengths)
        if maxsteplength >= maxstep:
            a1 *= maxstep / maxsteplength
            logger.debug('Rescaled a1 to fulfill maxstep criterion')
        self.a_start = a1
        logger.debug('phi_start = %e, phi_prime_start = %e', self.func_start, self.phi_prime_start)
        logger.debug('func_start = %s, self.func_old = %s', self.func_start, self.func_old)
        logger.debug('a1 = %e, a_max = %e, a_min = %e', a1, a_max, self.a_min)
        return a1