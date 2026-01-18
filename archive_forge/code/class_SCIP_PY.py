import operator
import os
import sys
import warnings
from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError
from .core import scip_path, fscip_path
from .. import constants
from typing import Dict, List, Optional, Tuple
class SCIP_PY(LpSolver):
    """
    The SCIP Optimization Suite (via its python interface)

    The SCIP internals are available after calling solve as:
    - each variable in variable.solverVar
    - each constraint in constraint.solverConstraint
    - the model in problem.solverModel
    """
    name = 'SCIP_PY'
    try:
        global scip
        import pyscipopt as scip
    except ImportError:

        def available(self):
            """True if the solver is available"""
            return False

        def actualSolve(self, lp):
            """Solve a well formulated lp problem"""
            raise PulpSolverError(f'The {self.name} solver is not available')
    else:

        def __init__(self, mip=True, msg=True, options=None, timeLimit=None, gapRel=None, gapAbs=None, maxNodes=None, logPath=None, threads=None, warmStart=False):
            """
            :param bool mip: if False, assume LP even if integer variables
            :param bool msg: if False, no log is shown
            :param list options: list of additional options to pass to solver
            :param float timeLimit: maximum time for solver (in seconds)
            :param float gapRel: relative gap tolerance for the solver to stop (in fraction)
            :param float gapAbs: absolute gap tolerance for the solver to stop
            :param int maxNodes: max number of nodes during branching. Stops the solving when reached.
            :param str logPath: path to the log file
            :param int threads: sets the maximum number of threads
            :param bool warmStart: if True, the solver will use the current value of variables as a start
            """
            super().__init__(mip=mip, msg=msg, options=options, timeLimit=timeLimit, gapRel=gapRel, gapAbs=gapAbs, maxNodes=maxNodes, logPath=logPath, threads=threads, warmStart=warmStart)

        def findSolutionValues(self, lp):
            lp.resolveOK = True
            solutionStatus = lp.solverModel.getStatus()
            scip_to_pulp_status = {'optimal': constants.LpStatusOptimal, 'unbounded': constants.LpStatusUnbounded, 'infeasible': constants.LpStatusInfeasible, 'inforunbd': constants.LpStatusNotSolved, 'timelimit': constants.LpStatusNotSolved, 'userinterrupt': constants.LpStatusNotSolved, 'nodelimit': constants.LpStatusNotSolved, 'totalnodelimit': constants.LpStatusNotSolved, 'stallnodelimit': constants.LpStatusNotSolved, 'gaplimit': constants.LpStatusNotSolved, 'memlimit': constants.LpStatusNotSolved, 'sollimit': constants.LpStatusNotSolved, 'bestsollimit': constants.LpStatusNotSolved, 'restartlimit': constants.LpStatusNotSolved, 'unknown': constants.LpStatusUndefined}
            possible_solution_found_statuses = ('optimal', 'timelimit', 'userinterrupt', 'nodelimit', 'totalnodelimit', 'stallnodelimit', 'gaplimit', 'memlimit')
            status = scip_to_pulp_status[solutionStatus]
            if solutionStatus in possible_solution_found_statuses:
                try:
                    solution = lp.solverModel.getBestSol()
                    for variable in lp._variables:
                        variable.varValue = solution[variable.solverVar]
                    for constraint in lp.constraints.values():
                        constraint.slack = lp.solverModel.getSlack(constraint.solverConstraint, solution)
                    if status == constants.LpStatusOptimal:
                        lp.assignStatus(status, constants.LpSolutionOptimal)
                    else:
                        status = constants.LpStatusOptimal
                        lp.assignStatus(status, constants.LpSolutionIntegerFeasible)
                except:
                    lp.assignStatus(status, constants.LpSolutionNoSolutionFound)
            else:
                lp.assignStatus(status)
            return status

        def available(self):
            """True if the solver is available"""
            return True

        def callSolver(self, lp):
            """Solves the problem with scip"""
            lp.solverModel.optimize()

        def buildSolverModel(self, lp):
            """
            Takes the pulp lp model and translates it into a scip model
            """
            lp.solverModel = scip.Model(lp.name)
            if lp.sense == constants.LpMaximize:
                lp.solverModel.setMaximize()
            else:
                lp.solverModel.setMinimize()
            if not self.msg:
                lp.solverModel.hideOutput()
            if self.timeLimit is not None:
                lp.solverModel.setParam('limits/time', self.timeLimit)
            if 'gapRel' in self.optionsDict:
                lp.solverModel.setParam('limits/gap', self.optionsDict['gapRel'])
            if 'gapAbs' in self.optionsDict:
                lp.solverModel.setParam('limits/absgap', self.optionsDict['gapAbs'])
            if 'maxNodes' in self.optionsDict:
                lp.solverModel.setParam('limits/nodes', self.optionsDict['maxNodes'])
            if 'logPath' in self.optionsDict:
                lp.solverModel.setLogfile(self.optionsDict['logPath'])
            if 'threads' in self.optionsDict and int(self.optionsDict['threads']) > 1:
                warnings.warn(f'The solver {self.name} can only run with a single thread')
            if not self.mip:
                warnings.warn(f'{self.name} does not allow a problem to be relaxed')
            options = iter(self.options)
            for option in options:
                if '=' in option:
                    name, value = option.split('=', maxsplit=2)
                else:
                    name, value = (option, next(options))
                lp.solverModel.setParam(name, value)
            category_to_vtype = {constants.LpBinary: 'B', constants.LpContinuous: 'C', constants.LpInteger: 'I'}
            for var in lp.variables():
                var.solverVar = lp.solverModel.addVar(name=var.name, vtype=category_to_vtype[var.cat], lb=var.lowBound, ub=var.upBound, obj=lp.objective.get(var, 0.0))
            sense_to_operator = {constants.LpConstraintLE: operator.le, constants.LpConstraintGE: operator.ge, constants.LpConstraintEQ: operator.eq}
            for name, constraint in lp.constraints.items():
                constraint.solverConstraint = lp.solverModel.addCons(cons=sense_to_operator[constraint.sense](scip.quicksum((coefficient * variable.solverVar for variable, coefficient in constraint.items())), -constraint.constant), name=name)
            if self.optionsDict.get('warmStart', False):
                s = lp.solverModel.createPartialSol()
                for var in lp.variables():
                    if var.varValue is not None:
                        lp.solverModel.setSolVal(s, var.solverVar, var.varValue)
                lp.solverModel.addSol(s)

        def actualSolve(self, lp):
            """
            Solve a well formulated lp problem

            creates a scip model, variables and constraints and attaches
            them to the lp model which it then solves
            """
            self.buildSolverModel(lp)
            self.callSolver(lp)
            solutionStatus = self.findSolutionValues(lp)
            for variable in lp._variables:
                variable.modified = False
            for constraint in lp.constraints.values():
                constraint.modified = False
            return solutionStatus

        def actualResolve(self, lp):
            """
            Solve a well formulated lp problem

            uses the old solver and modifies the rhs of the modified constraints
            """
            raise PulpSolverError(f'The {self.name} solver does not implement resolving')