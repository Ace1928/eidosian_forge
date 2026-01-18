from pyomo.common.config import (
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
from pyomo.core.expr import differentiate
from pyomo.common.collections import ComponentSet
from pyomo.opt import SolverFactory
from pyomo.repn import generate_standard_repn
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import (
from pyomo.contrib.fme.fourier_motzkin_elimination import (
import logging
Relax convex disjunctive model by forming the bigm relaxation and then
    iteratively adding cuts from the hull relaxation (or the hull relaxation
    after some basic steps) in order to strengthen the formulation.

    Note that gdp.cuttingplane is not a structural transformation: If variables
    on the model are fixed, they will be treated as data, and unfixing them
    after transformation will very likely result in an invalid model.

    This transformation accepts the following keyword arguments:

    Parameters
    ----------
    solver : Solver name (as string) to use to solve relaxed BigM and separation
             problems
    solver_options : dictionary of options to pass to the solver
    stream_solver : Whether or not to display solver output
    verbose : Enable verbose output from cuttingplanes algorithm
    cuts_name : Optional name for the IndexedConstraint containing the projected
                cuts (must be a unique name with respect to the instance)
    minimum_improvement_threshold : Stopping criterion based on improvement in
                                    Big-M relaxation. This is the minimum
                                    difference in relaxed BigM objective
                                    values between consecutive iterations
    separation_objective_threshold : Stopping criterion based on separation
                                     objective. If separation objective is not
                                     at least this large, cut generation will
                                     terminate.
    cut_filtering_threshold : Stopping criterion based on effectiveness of the
                              generated cut: This is the amount by which
                              a cut must be violated at the relaxed bigM
                              solution in order to be added to the bigM model
    max_number_of_cuts : The maximum number of cuts to add to the big-M model
    norm : norm to use in the objective of the separation problem
    tighten_relaxation : callback to modify the GDP model before the hull
                         relaxation is taken (e.g. could be used to perform
                         basic steps)
    create_cuts : callback to create cuts using the solved relaxed bigM and hull
                  problems
    post_process_cut : callback to perform post-processing on created cuts
    back_off_problem_tolerance : tolerance to use while post-processing
    zero_tolerance : Tolerance at which a float will be considered 0 when
                     using Fourier-Motzkin elimination to create cuts.
    do_integer_arithmetic : Whether or not to require Fourier-Motzkin elimination
                            to do integer arithmetic. Only possible when all
                            data is integer.
    tight_constraint_tolerance : Tolerance at which a constraint is considered
                                 tight for the Fourier-Motzkin cut generation
                                 procedure

    By default, the callbacks will be set such that the algorithm performed is
    as presented in [1], but with an additional post-processing procedure to
    reduce numerical error, which calculates the maximum violation of the cut
    subject to the relaxed hull constraints, and then pads the constraint by
    this violation plus an additional user-specified tolerance.

    In addition, the create_cuts_fme function provides an (exponential time)
    method of generating cuts which reduces numerical error (and can eliminate
    it if all data is integer). It collects the hull constraints which are
    tight at the solution of the separation problem. It creates a cut in the
    extended space perpendicular to  a composite normal vector created by
    summing the directions normal to these constraints. It then performs
    fourier-motzkin elimination on the collection of constraints and the cut
    to project out the disaggregated variables. The resulting constraint which
    is most violated by the relaxed bigM solution is then returned.

    References
    ----------
        [1] Sawaya, N. W., Grossmann, I. E. (2005). A cutting plane method for
        solving linear generalized disjunctive programming problems. Computers
        and Chemical Engineering, 29, 1891-1913
    