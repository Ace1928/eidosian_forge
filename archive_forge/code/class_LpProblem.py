from collections import Counter
import sys
import warnings
from time import time
from .apis import LpSolverDefault, PULP_CBC_CMD
from .apis.core import clock
from .utilities import value
from . import constants as const
from . import mps_lp as mpslp
import logging
import re
class LpProblem:
    """An LP Problem"""

    def __init__(self, name='NoName', sense=const.LpMinimize):
        """
        Creates an LP Problem

        This function creates a new LP Problem  with the specified associated parameters

        :param name: name of the problem used in the output .lp file
        :param sense: of the LP problem objective.                  Either :data:`~pulp.const.LpMinimize` (default)                 or :data:`~pulp.const.LpMaximize`.
        :return: An LP Problem
        """
        if ' ' in name:
            warnings.warn("Spaces are not permitted in the name. Converted to '_'")
            name = name.replace(' ', '_')
        self.objective = None
        self.constraints = _DICT_TYPE()
        self.name = name
        self.sense = sense
        self.sos1 = {}
        self.sos2 = {}
        self.status = const.LpStatusNotSolved
        self.sol_status = const.LpSolutionNoSolutionFound
        self.noOverlap = 1
        self.solver = None
        self.modifiedVariables = []
        self.modifiedConstraints = []
        self.resolveOK = False
        self._variables = []
        self._variable_ids = {}
        self.dummyVar = None
        self.solutionTime = 0
        self.solutionCpuTime = 0
        self.lastUnused = 0

    def __repr__(self):
        s = self.name + ':\n'
        if self.sense == 1:
            s += 'MINIMIZE\n'
        else:
            s += 'MAXIMIZE\n'
        s += repr(self.objective) + '\n'
        if self.constraints:
            s += 'SUBJECT TO\n'
            for n, c in self.constraints.items():
                s += c.asCplexLpConstraint(n) + '\n'
        s += 'VARIABLES\n'
        for v in self.variables():
            s += v.asCplexLpVariable() + ' ' + const.LpCategories[v.cat] + '\n'
        return s

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_variable_ids']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._variable_ids = {}
        for v in self._variables:
            self._variable_ids[v.hash] = v

    def copy(self):
        """Make a copy of self. Expressions are copied by reference"""
        lpcopy = LpProblem(name=self.name, sense=self.sense)
        lpcopy.objective = self.objective
        lpcopy.constraints = self.constraints.copy()
        lpcopy.sos1 = self.sos1.copy()
        lpcopy.sos2 = self.sos2.copy()
        return lpcopy

    def deepcopy(self):
        """Make a copy of self. Expressions are copied by value"""
        lpcopy = LpProblem(name=self.name, sense=self.sense)
        if self.objective is not None:
            lpcopy.objective = self.objective.copy()
        lpcopy.constraints = {}
        for k, v in self.constraints.items():
            lpcopy.constraints[k] = v.copy()
        lpcopy.sos1 = self.sos1.copy()
        lpcopy.sos2 = self.sos2.copy()
        return lpcopy

    def toDict(self):
        """
        creates a dictionary from the model with as much data as possible.
        It replaces variables by variable names.
        So it requires to have unique names for variables.

        :return: dictionary with model data
        :rtype: dict
        """
        try:
            self.checkDuplicateVars()
        except const.PulpError:
            raise const.PulpError('Duplicated names found in variables:\nto export the model, variable names need to be unique')
        self.fixObjective()
        variables = self.variables()
        return dict(objective=dict(name=self.objective.name, coefficients=self.objective.toDict()), constraints=[v.toDict() for v in self.constraints.values()], variables=[v.toDict() for v in variables], parameters=dict(name=self.name, sense=self.sense, status=self.status, sol_status=self.sol_status), sos1=list(self.sos1.values()), sos2=list(self.sos2.values()))
    to_dict = toDict

    @classmethod
    def fromDict(cls, _dict):
        """
        Takes a dictionary with all necessary information to build a model.
        And returns a dictionary of variables and a problem object

        :param _dict: dictionary with the model stored
        :return: a tuple with a dictionary of variables and a :py:class:`LpProblem`
        """
        params = _dict['parameters']
        pb_params = {'name', 'sense'}
        args = {k: params[k] for k in pb_params}
        pb = cls(**args)
        pb.status = params['status']
        pb.sol_status = params['sol_status']
        var = {v['name']: LpVariable.fromDict(**v) for v in _dict['variables']}
        obj_e = {var[v['name']]: v['value'] for v in _dict['objective']['coefficients']}
        pb += LpAffineExpression(e=obj_e, name=_dict['objective']['name'])

        def edit_const(const):
            const = dict(const)
            const['coefficients'] = {var[v['name']]: v['value'] for v in const['coefficients']}
            return const
        constraints = [edit_const(v) for v in _dict['constraints']]
        for c in constraints:
            pb += LpConstraint.fromDict(c)
        list_to_dict = lambda v: {k: v for k, v in enumerate(v)}
        pb.sos1 = list_to_dict(_dict['sos1'])
        pb.sos2 = list_to_dict(_dict['sos2'])
        return (var, pb)
    from_dict = fromDict

    def toJson(self, filename, *args, **kwargs):
        """
        Creates a json file from the LpProblem information

        :param str filename: filename to write json
        :param args: additional arguments for json function
        :param kwargs: additional keyword arguments for json function
        :return: None
        """
        with open(filename, 'w') as f:
            json.dump(self.toDict(), f, *args, **kwargs)
    to_json = toJson

    @classmethod
    def fromJson(cls, filename):
        """
        Creates a new Lp Problem from a json file with information

        :param str filename: json file name
        :return: a tuple with a dictionary of variables and an LpProblem
        :rtype: (dict, :py:class:`LpProblem`)
        """
        with open(filename) as f:
            data = json.load(f)
        return cls.fromDict(data)
    from_json = fromJson

    @classmethod
    def fromMPS(cls, filename, sense=const.LpMinimize, **kwargs):
        data = mpslp.readMPS(filename, sense=sense, **kwargs)
        return cls.fromDict(data)

    def normalisedNames(self):
        constraintsNames = {k: 'C%07d' % i for i, k in enumerate(self.constraints)}
        _variables = self.variables()
        variablesNames = {k.name: 'X%07d' % i for i, k in enumerate(_variables)}
        return (constraintsNames, variablesNames, 'OBJ')

    def isMIP(self):
        for v in self.variables():
            if v.cat == const.LpInteger:
                return 1
        return 0

    def roundSolution(self, epsInt=1e-05, eps=1e-07):
        """
        Rounds the lp variables

        Inputs:
            - none

        Side Effects:
            - The lp variables are rounded
        """
        for v in self.variables():
            v.round(epsInt, eps)

    def unusedConstraintName(self):
        self.lastUnused += 1
        while 1:
            s = '_C%d' % self.lastUnused
            if s not in self.constraints:
                break
            self.lastUnused += 1
        return s

    def valid(self, eps=0):
        for v in self.variables():
            if not v.valid(eps):
                return False
        for c in self.constraints.values():
            if not c.valid(eps):
                return False
        else:
            return True

    def infeasibilityGap(self, mip=1):
        gap = 0
        for v in self.variables():
            gap = max(abs(v.infeasibilityGap(mip)), gap)
        for c in self.constraints.values():
            if not c.valid(0):
                gap = max(abs(c.value()), gap)
        return gap

    def addVariable(self, variable):
        """
        Adds a variable to the problem before a constraint is added

        @param variable: the variable to be added
        """
        if variable.hash not in self._variable_ids:
            self._variables.append(variable)
            self._variable_ids[variable.hash] = variable

    def addVariables(self, variables):
        """
        Adds variables to the problem before a constraint is added

        @param variables: the variables to be added
        """
        for v in variables:
            self.addVariable(v)

    def variables(self):
        """
        Returns the problem variables

        :return: A list containing the problem variables
        :rtype: (list, :py:class:`LpVariable`)
        """
        if self.objective:
            self.addVariables(list(self.objective.keys()))
        for c in self.constraints.values():
            self.addVariables(list(c.keys()))
        self._variables.sort(key=lambda v: v.name)
        return self._variables

    def variablesDict(self):
        variables = {}
        if self.objective:
            for v in self.objective:
                variables[v.name] = v
        for c in list(self.constraints.values()):
            for v in c:
                variables[v.name] = v
        return variables

    def add(self, constraint, name=None):
        self.addConstraint(constraint, name)

    def addConstraint(self, constraint, name=None):
        if not isinstance(constraint, LpConstraint):
            raise TypeError('Can only add LpConstraint objects')
        if name:
            constraint.name = name
        try:
            if constraint.name:
                name = constraint.name
            else:
                name = self.unusedConstraintName()
        except AttributeError:
            raise TypeError('Can only add LpConstraint objects')
        if name in self.constraints:
            if self.noOverlap:
                raise const.PulpError('overlapping constraint names: ' + name)
            else:
                print('Warning: overlapping constraint names:', name)
        self.constraints[name] = constraint
        self.modifiedConstraints.append(constraint)
        self.addVariables(list(constraint.keys()))

    def setObjective(self, obj):
        """
        Sets the input variable as the objective function. Used in Columnwise Modelling

        :param obj: the objective function of type :class:`LpConstraintVar`

        Side Effects:
            - The objective function is set
        """
        if isinstance(obj, LpVariable):
            obj = obj + 0.0
        try:
            obj = obj.constraint
            name = obj.name
        except AttributeError:
            name = None
        self.objective = obj
        self.objective.name = name
        self.resolveOK = False

    def __iadd__(self, other):
        if isinstance(other, tuple):
            other, name = other
        else:
            name = None
        if other is True:
            return self
        elif other is False:
            raise TypeError('A False object cannot be passed as a constraint')
        elif isinstance(other, LpConstraintVar):
            self.addConstraint(other.constraint)
        elif isinstance(other, LpConstraint):
            self.addConstraint(other, name)
        elif isinstance(other, LpAffineExpression):
            if self.objective is not None:
                warnings.warn('Overwriting previously set objective.')
            self.objective = other
            if name is not None:
                self.objective.name = name
        elif isinstance(other, LpVariable) or isinstance(other, (int, float)):
            if self.objective is not None:
                warnings.warn('Overwriting previously set objective.')
            self.objective = LpAffineExpression(other)
            self.objective.name = name
        else:
            raise TypeError('Can only add LpConstraintVar, LpConstraint, LpAffineExpression or True objects')
        return self

    def extend(self, other, use_objective=True):
        """
        extends an LpProblem by adding constraints either from a dictionary
        a tuple or another LpProblem object.

        @param use_objective: determines whether the objective is imported from
        the other problem

        For dictionaries the constraints will be named with the keys
        For tuples an unique name will be generated
        For LpProblems the name of the problem will be added to the constraints
        name
        """
        if isinstance(other, dict):
            for name in other:
                self.constraints[name] = other[name]
        elif isinstance(other, LpProblem):
            for v in set(other.variables()).difference(self.variables()):
                v.name = other.name + v.name
            for name, c in other.constraints.items():
                c.name = other.name + name
                self.addConstraint(c)
            if use_objective:
                self.objective += other.objective
        else:
            for c in other:
                if isinstance(c, tuple):
                    name = c[0]
                    c = c[1]
                else:
                    name = None
                if not name:
                    name = c.name
                if not name:
                    name = self.unusedConstraintName()
                self.constraints[name] = c

    def coefficients(self, translation=None):
        coefs = []
        if translation == None:
            for c in self.constraints:
                cst = self.constraints[c]
                coefs.extend([(v.name, c, cst[v]) for v in cst])
        else:
            for c in self.constraints:
                ctr = translation[c]
                cst = self.constraints[c]
                coefs.extend([(translation[v.name], ctr, cst[v]) for v in cst])
        return coefs

    def writeMPS(self, filename, mpsSense=0, rename=0, mip=1, with_objsense: bool=False):
        """
        Writes an mps files from the problem information

        :param str filename: name of the file to write
        :param int mpsSense:
        :param bool rename: if True, normalized names are used for variables and constraints
        :param mip: variables and variable renames
        :return:
        Side Effects:
            - The file is created
        """
        return mpslp.writeMPS(self, filename, mpsSense=mpsSense, rename=rename, mip=mip, with_objsense=with_objsense)

    def writeLP(self, filename, writeSOS=1, mip=1, max_length=100):
        """
        Write the given Lp problem to a .lp file.

        This function writes the specifications (objective function,
        constraints, variables) of the defined Lp problem to a file.

        :param str filename: the name of the file to be created.
        :return: variables
        Side Effects:
            - The file is created
        """
        return mpslp.writeLP(self, filename=filename, writeSOS=writeSOS, mip=mip, max_length=max_length)

    def checkDuplicateVars(self) -> None:
        """
        Checks if there are at least two variables with the same name
        :return: 1
        :raises `const.PulpError`: if there ar duplicates
        """
        name_counter = Counter((variable.name for variable in self.variables()))
        repeated_names = {(name, count) for name, count in name_counter.items() if count >= 2}
        if repeated_names:
            raise const.PulpError(f'Repeated variable names: {repeated_names}')

    def checkLengthVars(self, max_length: int) -> None:
        """
        Checks if variables have names smaller than `max_length`
        :param int max_length: max size for variable name
        :return:
        :raises const.PulpError: if there is at least one variable that has a long name
        """
        long_names = [variable.name for variable in self.variables() if len(variable.name) > max_length]
        if long_names:
            raise const.PulpError(f'Variable names too long for Lp format: {long_names}')

    def assignVarsVals(self, values):
        variables = self.variablesDict()
        for name in values:
            if name != '__dummy':
                variables[name].varValue = values[name]

    def assignVarsDj(self, values):
        variables = self.variablesDict()
        for name in values:
            if name != '__dummy':
                variables[name].dj = values[name]

    def assignConsPi(self, values):
        for name in values:
            try:
                self.constraints[name].pi = values[name]
            except KeyError:
                pass

    def assignConsSlack(self, values, activity=False):
        for name in values:
            try:
                if activity:
                    self.constraints[name].slack = -1 * (self.constraints[name].constant + float(values[name]))
                else:
                    self.constraints[name].slack = float(values[name])
            except KeyError:
                pass

    def get_dummyVar(self):
        if self.dummyVar is None:
            self.dummyVar = LpVariable('__dummy', 0, 0)
        return self.dummyVar

    def fixObjective(self):
        if self.objective is None:
            self.objective = 0
            wasNone = 1
        else:
            wasNone = 0
        if not isinstance(self.objective, LpAffineExpression):
            self.objective = LpAffineExpression(self.objective)
        if self.objective.isNumericalConstant():
            dummyVar = self.get_dummyVar()
            self.objective += dummyVar
        else:
            dummyVar = None
        return (wasNone, dummyVar)

    def restoreObjective(self, wasNone, dummyVar):
        if wasNone:
            self.objective = None
        elif not dummyVar is None:
            self.objective -= dummyVar

    def solve(self, solver=None, **kwargs):
        """
        Solve the given Lp problem.

        This function changes the problem to make it suitable for solving
        then calls the solver.actualSolve() method to find the solution

        :param solver:  Optional: the specific solver to be used, defaults to the
              default solver.

        Side Effects:
            - The attributes of the problem object are changed in
              :meth:`~pulp.solver.LpSolver.actualSolve()` to reflect the Lp solution
        """
        if not solver:
            solver = self.solver
        if not solver:
            solver = LpSolverDefault
        wasNone, dummyVar = self.fixObjective()
        self.startClock()
        status = solver.actualSolve(self, **kwargs)
        self.stopClock()
        self.restoreObjective(wasNone, dummyVar)
        self.solver = solver
        return status

    def startClock(self):
        """initializes properties with the current time"""
        self.solutionCpuTime = -clock()
        self.solutionTime = -time()

    def stopClock(self):
        """updates time wall time and cpu time"""
        self.solutionTime += time()
        self.solutionCpuTime += clock()

    def sequentialSolve(self, objectives, absoluteTols=None, relativeTols=None, solver=None, debug=False):
        """
        Solve the given Lp problem with several objective functions.

        This function sequentially changes the objective of the problem
        and then adds the objective function as a constraint

        :param objectives: the list of objectives to be used to solve the problem
        :param absoluteTols: the list of absolute tolerances to be applied to
           the constraints should be +ve for a minimise objective
        :param relativeTols: the list of relative tolerances applied to the constraints
        :param solver: the specific solver to be used, defaults to the default solver.

        """
        if not solver:
            solver = self.solver
        if not solver:
            solver = LpSolverDefault
        if not absoluteTols:
            absoluteTols = [0] * len(objectives)
        if not relativeTols:
            relativeTols = [1] * len(objectives)
        self.startClock()
        statuses = []
        for i, (obj, absol, rel) in enumerate(zip(objectives, absoluteTols, relativeTols)):
            self.setObjective(obj)
            status = solver.actualSolve(self)
            statuses.append(status)
            if debug:
                self.writeLP(f'{i}Sequence.lp')
            if self.sense == const.LpMinimize:
                self += (obj <= value(obj) * rel + absol, f'Sequence_Objective_{i}')
            elif self.sense == const.LpMaximize:
                self += (obj >= value(obj) * rel + absol, f'Sequence_Objective_{i}')
        self.stopClock()
        self.solver = solver
        return statuses

    def resolve(self, solver=None, **kwargs):
        """
        resolves an Problem using the same solver as previously
        """
        if not solver:
            solver = self.solver
        if self.resolveOK:
            return self.solver.actualResolve(self, **kwargs)
        else:
            return self.solve(solver=solver, **kwargs)

    def setSolver(self, solver=LpSolverDefault):
        """Sets the Solver for this problem useful if you are using
        resolve
        """
        self.solver = solver

    def numVariables(self):
        """

        :return: number of variables in model
        """
        return len(self._variable_ids)

    def numConstraints(self):
        """

        :return: number of constraints in model
        """
        return len(self.constraints)

    def getSense(self):
        return self.sense

    def assignStatus(self, status, sol_status=None):
        """
        Sets the status of the model after solving.
        :param status: code for the status of the model
        :param sol_status: code for the status of the solution
        :return:
        """
        if status not in const.LpStatus:
            raise const.PulpError('Invalid status code: ' + str(status))
        if sol_status is not None and sol_status not in const.LpSolution:
            raise const.PulpError('Invalid solution status code: ' + str(sol_status))
        self.status = status
        if sol_status is None:
            sol_status = const.LpStatusToSolution.get(status, const.LpSolutionNoSolutionFound)
        self.sol_status = sol_status
        return True