from math import inf
from typing import List
from .core import LpSolver, LpSolver_CMD, subprocess, PulpSolverError
import os, sys
from .. import constants
class HiGHS(LpSolver):
    name = 'HiGHS'
    try:
        global highspy
        import highspy
    except:

        def available(self):
            """True if the solver is available"""
            return False

        def actualSolve(self, lp, callback=None):
            """Solve a well formulated lp problem"""
            raise PulpSolverError('HiGHS: Not Available')
    else:
        DEFAULT_CALLBACK = lambda logType, logMsg, callbackValue: print(f'[{logType.name}] {logMsg}')
        DEFAULT_CALLBACK_VALUE = ''

        def __init__(self, mip=True, msg=True, callbackTuple=None, gapAbs=None, gapRel=None, threads=None, timeLimit=None, **solverParams):
            """
            :param bool mip: if False, assume LP even if integer variables
            :param bool msg: if False, no log is shown
            :param tuple callbackTuple: Tuple of log callback function (see DEFAULT_CALLBACK above for definition)
                and callbackValue (tag embedded in every callback)
            :param float gapRel: relative gap tolerance for the solver to stop (in fraction)
            :param float gapAbs: absolute gap tolerance for the solver to stop
            :param int threads: sets the maximum number of threads
            :param float timeLimit: maximum time for solver (in seconds)
            :param dict solverParams: list of named options to pass directly to the HiGHS solver
            """
            super().__init__(mip=mip, msg=msg, timeLimit=timeLimit, **solverParams)
            self.callbackTuple = callbackTuple
            self.gapAbs = gapAbs
            self.gapRel = gapRel
            self.threads = threads

        def available(self):
            return True

        def callSolver(self, lp):
            lp.solverModel.run()

        def createAndConfigureSolver(self, lp):
            lp.solverModel = highspy.Highs()
            if self.msg or self.callbackTuple:
                callbackTuple = self.callbackTuple or (HiGHS.DEFAULT_CALLBACK, HiGHS.DEFAULT_CALLBACK_VALUE)
                lp.solverModel.setLogCallback(*callbackTuple)
            if self.gapRel is not None:
                lp.solverModel.setOptionValue('mip_rel_gap', self.gapRel)
            if self.gapAbs is not None:
                lp.solverModel.setOptionValue('mip_abs_gap', self.gapAbs)
            if self.threads is not None:
                lp.solverModel.setOptionValue('threads', self.threads)
            if self.timeLimit is not None:
                lp.solverModel.setOptionValue('time_limit', float(self.timeLimit))
            for key, value in self.optionsDict.items():
                lp.solverModel.setOptionValue(key, value)

        def buildSolverModel(self, lp):
            inf = highspy.kHighsInf
            obj_mult = -1 if lp.sense == constants.LpMaximize else 1
            for i, var in enumerate(lp.variables()):
                lb = var.lowBound
                ub = var.upBound
                lp.solverModel.addCol(obj_mult * lp.objective.get(var, 0.0), -inf if lb is None else lb, inf if ub is None else ub, 0, [], [])
                var.index = i
                if var.cat == constants.LpInteger and self.mip:
                    lp.solverModel.changeColIntegrality(var.index, highspy.HighsVarType.kInteger)
            for constraint in lp.constraints.values():
                non_zero_constraint_items = [(var.index, coefficient) for var, coefficient in constraint.items() if coefficient != 0]
                if len(non_zero_constraint_items) == 0:
                    indices, coefficients = ([], [])
                else:
                    indices, coefficients = zip(*non_zero_constraint_items)
                lb = constraint.getLb()
                ub = constraint.getUb()
                lp.solverModel.addRow(-inf if lb is None else lb, inf if ub is None else ub, len(indices), indices, coefficients)

        def findSolutionValues(self, lp):
            status = lp.solverModel.getModelStatus()
            obj_value = lp.solverModel.getObjectiveValue()
            solution = lp.solverModel.getSolution()
            HighsModelStatus = highspy.HighsModelStatus
            status_dict = {HighsModelStatus.kNotset: (constants.LpStatusNotSolved, constants.LpSolutionNoSolutionFound), HighsModelStatus.kLoadError: (constants.LpStatusNotSolved, constants.LpSolutionNoSolutionFound), HighsModelStatus.kModelError: (constants.LpStatusNotSolved, constants.LpSolutionNoSolutionFound), HighsModelStatus.kPresolveError: (constants.LpStatusNotSolved, constants.LpSolutionNoSolutionFound), HighsModelStatus.kSolveError: (constants.LpStatusNotSolved, constants.LpSolutionNoSolutionFound), HighsModelStatus.kPostsolveError: (constants.LpStatusNotSolved, constants.LpSolutionNoSolutionFound), HighsModelStatus.kModelEmpty: (constants.LpStatusNotSolved, constants.LpSolutionNoSolutionFound), HighsModelStatus.kOptimal: (constants.LpStatusOptimal, constants.LpSolutionOptimal), HighsModelStatus.kInfeasible: (constants.LpStatusInfeasible, constants.LpSolutionInfeasible), HighsModelStatus.kUnboundedOrInfeasible: (constants.LpStatusInfeasible, constants.LpSolutionInfeasible), HighsModelStatus.kUnbounded: (constants.LpStatusUnbounded, constants.LpSolutionUnbounded), HighsModelStatus.kObjectiveBound: (constants.LpStatusOptimal, constants.LpSolutionIntegerFeasible), HighsModelStatus.kObjectiveTarget: (constants.LpStatusOptimal, constants.LpSolutionIntegerFeasible), HighsModelStatus.kTimeLimit: (constants.LpStatusOptimal, constants.LpSolutionIntegerFeasible), HighsModelStatus.kIterationLimit: (constants.LpStatusOptimal, constants.LpSolutionIntegerFeasible), HighsModelStatus.kUnknown: (constants.LpStatusNotSolved, constants.LpSolutionNoSolutionFound)}
            col_values = list(solution.col_value)
            for var in lp.variables():
                var.varValue = col_values[var.index]
            if obj_value == float(inf) and status in (HighsModelStatus.kTimeLimit, HighsModelStatus.kIterationLimit):
                return (constants.LpStatusNotSolved, constants.LpSolutionNoSolutionFound)
            else:
                return status_dict[status]

        def actualSolve(self, lp):
            self.createAndConfigureSolver(lp)
            self.buildSolverModel(lp)
            self.callSolver(lp)
            status, sol_status = self.findSolutionValues(lp)
            for var in lp.variables():
                var.modified = False
            for constraint in lp.constraints.values():
                constraint.modifier = False
            lp.assignStatus(status, sol_status)
            return status

        def actualResolve(self, lp, **kwargs):
            raise PulpSolverError('HiGHS: Resolving is not supported')