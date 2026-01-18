from .core import LpSolver, LpSolver_CMD, subprocess, PulpSolverError
from .. import constants
import warnings
import sys
import re
class XPRESS_PY(LpSolver):
    """The XPRESS LP solver that uses XPRESS Python API"""
    name = 'XPRESS_PY'

    def __init__(self, mip=True, msg=True, timeLimit=None, gapRel=None, heurFreq=None, heurStra=None, coverCuts=None, preSolve=None, warmStart=None, export=None, options=None):
        """
        Initializes the Xpress solver.

        :param bool mip: if False, assume LP even if integer variables
        :param bool msg: if False, no log is shown
        :param float timeLimit: maximum time for solver (in seconds)
        :param float gapRel: relative gap tolerance for the solver to stop (in fraction)
        :param heurFreq: the frequency at which heuristics are used in the tree search
        :param heurStra: heuristic strategy
        :param coverCuts: the number of rounds of lifted cover inequalities at the top node
        :param preSolve: whether presolving should be performed before the main algorithm
        :param bool warmStart: if set then use current variable values as warm start
        :param string export: if set then the model will be exported to this file before solving
        :param options: Adding more options. This is a list the elements of which
                        are either (name,value) pairs or strings "name=value".
                        More about Xpress options and control parameters please see
                        https://www.fico.com/fico-xpress-optimization/docs/latest/solver/optimizer/HTML/chapter7.html
        """
        if timeLimit is not None:
            timeLimit = -abs(timeLimit)
        LpSolver.__init__(self, gapRel=gapRel, mip=mip, msg=msg, timeLimit=timeLimit, options=options, heurFreq=heurFreq, heurStra=heurStra, coverCuts=coverCuts, preSolve=preSolve, warmStart=warmStart)
        self._available = None
        self._export = export

    def available(self):
        """True if the solver is available"""
        if self._available is None:
            try:
                global xpress
                import xpress
                xpress.setOutputEnabled(False)
                self._available = True
            except:
                self._available = False
        return self._available

    def callSolver(self, lp, prepare=None):
        """Perform the actual solve from actualSolve() or actualResolve().

        :param prepare:  a function that is called with `lp` as argument
                         and allows final tweaks to `lp.solverModel` before
                         the low level solve is started.
        """
        try:
            model = lp.solverModel
            for v in lp.variables():
                v.modified = False
            for c in lp.constraints.values():
                c.modified = False
            if self._export is not None:
                if self._export.lower().endswith('.lp'):
                    model.write(self._export, 'l')
                else:
                    model.write(self._export)
            if prepare is not None:
                prepare(lp)
            if _ismip(lp) and (not self.mip):
                model.lpoptimize()
            else:
                model.solve()
        except (xpress.ModelError, xpress.InterfaceError, xpress.SolverError) as err:
            raise PulpSolverError(str(err))

    def findSolutionValues(self, lp):
        try:
            model = lp.solverModel
            if _ismip(lp) and self.mip:
                x, slacks, duals, djs = ([], [], None, None)
                try:
                    model.getmipsol(x, slacks)
                except:
                    x, slacks = (None, None)
                statusmap = {0: constants.LpStatusUndefined, 1: constants.LpStatusUndefined, 2: constants.LpStatusUndefined, 3: constants.LpStatusUndefined, 4: constants.LpStatusUndefined, 5: constants.LpStatusInfeasible, 6: constants.LpStatusOptimal, 7: constants.LpStatusUndefined}
                statuskey = 'mipstatus'
            else:
                x, slacks, duals, djs = ([], [], [], [])
                try:
                    model.getlpsol(x, slacks, duals, djs)
                except:
                    x, slacks, duals, djs = (None, None, None, None)
                statusmap = {0: constants.LpStatusNotSolved, 1: constants.LpStatusOptimal, 2: constants.LpStatusInfeasible, 3: constants.LpStatusUndefined, 4: constants.LpStatusUndefined, 5: constants.LpStatusUnbounded, 6: constants.LpStatusUndefined, 7: constants.LpStatusNotSolved, 8: constants.LpStatusUndefined}
                statuskey = 'lpstatus'
            if x is not None:
                lp.assignVarsVals({v.name: x[v._xprs[0]] for v in lp.variables()})
            if djs is not None:
                lp.assignVarsDj({v.name: djs[v._xprs[0]] for v in lp.variables()})
            if duals is not None:
                lp.assignConsPi({c.name: duals[c._xprs[0]] for c in lp.constraints.values()})
            if slacks is not None:
                lp.assignConsSlack({c.name: slacks[c._xprs[0]] for c in lp.constraints.values()})
            status = statusmap.get(model.getAttrib(statuskey), constants.LpStatusUndefined)
            lp.assignStatus(status)
            return status
        except (xpress.ModelError, xpress.InterfaceError, xpress.SolverError) as err:
            raise PulpSolverError(str(err))

    def actualSolve(self, lp, prepare=None):
        """Solve a well formulated lp problem"""
        if not self.available():
            message = 'XPRESS Python API not available'
            try:
                import xpress
            except ImportError as err:
                message = str(err)
            raise PulpSolverError(message)
        self.buildSolverModel(lp)
        self.callSolver(lp, prepare)
        return self.findSolutionValues(lp)

    def buildSolverModel(self, lp):
        """
        Takes the pulp lp model and translates it into an xpress model
        """
        self._extract(lp)
        try:
            model = lp.solverModel
            for key, name in [('gapRel', 'MIPRELSTOP'), ('timeLimit', 'MAXTIME'), ('heurFreq', 'HEURFREQ'), ('heurStra', 'HEURSTRATEGY'), ('coverCuts', 'COVERCUTS'), ('preSolve', 'PRESOLVE')]:
                value = self.optionsDict.get(key, None)
                if value is not None:
                    model.setControl(name, value)
            for option in self.options:
                if isinstance(option, tuple):
                    name = optione[0]
                    value = option[1]
                else:
                    fields = option.split('=', 1)
                    if len(fields) != 2:
                        raise PulpSolverError('Invalid option ' + str(option))
                    name = fields[0].strip()
                    value = fields[1].strip()
                try:
                    model.setControl(name, int(value))
                    continue
                except ValueError:
                    pass
                try:
                    model.setControl(name, float(value))
                    continue
                except ValueError:
                    pass
                model.setControl(name, value)
            if self.optionsDict.get('warmStart', False):
                solval = list()
                colind = list()
                for v in sorted(lp.variables(), key=lambda x: x._xprs[0]):
                    if v.value() is not None:
                        solval.append(v.value())
                        colind.append(v._xprs[0])
                if _ismip(lp) and self.mip:
                    if len(solval) == model.attributes.cols:
                        model.loadmipsol(solval)
                    else:
                        model.addmipsol(solval, colind, 'warmstart')
                else:
                    model.loadlpsol(solval, None, None, None)
            if self.msg:

                def message(prob, data, msg, msgtype):
                    if msgtype > 0:
                        print(msg)
                model.addcbmessage(message)
        except (xpress.ModelError, xpress.InterfaceError, xpress.SolverError) as err:
            raise PulpSolverError(str(err))

    def actualResolve(self, lp, prepare=None):
        """Resolve a problem that was previously solved by actualSolve()."""
        try:
            rhsind = list()
            rhsval = list()
            for name in sorted(lp.constraints):
                con = lp.constraints[name]
                if not con.modified:
                    continue
                if not hasattr(con, '_xprs'):
                    raise PulpSolverError('Cannot add new constraints')
                rhsind.append(con._xprs[0])
                rhsval.append(-con.constant)
            if len(rhsind) > 0:
                lp.solverModel.chgrhs(rhsind, rhsval)
            bndind = list()
            bndtype = list()
            bndval = list()
            for v in lp.variables():
                if not v.modified:
                    continue
                if not hasattr(v, '_xprs'):
                    raise PulpSolverError('Cannot add new variables')
                bndind.append(v._xprs[0])
                bndtype.append('L')
                bndval.append(-xpress.infinity if v.lowBound is None else v.lowBound)
                bndind.append(v._xprs[0])
                bndtype.append('G')
                bndval.append(xpress.infinity if v.upBound is None else v.upBound)
            if len(bndtype) > 0:
                lp.solverModel.chgbounds(bndind, bndtype, bndval)
            self.callSolver(lp, prepare)
            return self.findSolutionValues(lp)
        except (xpress.ModelError, xpress.InterfaceError, xpress.SolverError) as err:
            raise PulpSolverError(str(err))

    @staticmethod
    def _reset(lp):
        """Reset any XPRESS specific information in lp."""
        if hasattr(lp, 'solverModel'):
            delattr(lp, 'solverModel')
        for v in lp.variables():
            if hasattr(v, '_xprs'):
                delattr(v, '_xprs')
        for c in lp.constraints.values():
            if hasattr(c, '_xprs'):
                delattr(c, '_xprs')

    def _extract(self, lp):
        """Extract a given model to an XPRESS Python API instance.

        The function stores XPRESS specific information in the `solverModel` property
        of `lp` and each variable and constraint. These information can be
        removed by calling `_reset`.
        """
        self._reset(lp)
        try:
            model = xpress.problem()
            if lp.sense == constants.LpMaximize:
                model.chgobjsense(xpress.maximize)
            obj = list()
            lb = list()
            ub = list()
            ctype = list()
            names = list()
            for v in lp.variables():
                lb.append(-xpress.infinity if v.lowBound is None else v.lowBound)
                ub.append(xpress.infinity if v.upBound is None else v.upBound)
                obj.append(lp.objective.get(v, 0.0))
                if v.cat == constants.LpInteger:
                    ctype.append('I')
                elif v.cat == constants.LpBinary:
                    ctype.append('B')
                else:
                    ctype.append('C')
                names.append(v.name)
            model.addcols(obj, [0] * (len(obj) + 1), [], [], lb, ub, names, ctype)
            for j, (v, x) in enumerate(zip(lp.variables(), model.getVariable())):
                v._xprs = (j, x)
            cons = list()
            for i, name in enumerate(sorted(lp.constraints)):
                con = lp.constraints[name]
                lhs = xpress.Sum((a * x._xprs[1] for x, a in sorted(con.items(), key=lambda x: x[0]._xprs[0])))
                rhs = -con.constant
                if con.sense == constants.LpConstraintLE:
                    c = xpress.constraint(body=lhs, sense=xpress.leq, rhs=rhs)
                elif con.sense == constants.LpConstraintGE:
                    c = xpress.constraint(body=lhs, sense=xpress.geq, rhs=rhs)
                elif con.sense == constants.LpConstraintEQ:
                    c = xpress.constraint(body=lhs, sense=xpress.eq, rhs=rhs)
                else:
                    raise PulpSolverError('Unsupprted constraint type ' + str(con.sense))
                cons.append((i, c, con))
                if len(cons) > 100:
                    model.addConstraint([c for _, c, _ in cons])
                    for i, c, con in cons:
                        con._xprs = (i, c)
                    cons = list()
            if len(cons) > 0:
                model.addConstraint([c for _, c, _ in cons])
                for i, c, con in cons:
                    con._xprs = (i, c)

            def addsos(m, sosdict, sostype):
                """Extract sos constraints from PuLP."""
                soslist = []
                for name in sorted(sosdict):
                    indices = []
                    weights = []
                    for v, val in sosdict[name].items():
                        indices.append(v._xprs[0])
                        weights.append(val)
                    soslist.append(xpress.sos(indices, weights, sostype, str(name)))
                if len(soslist):
                    m.addSOS(soslist)
            addsos(model, lp.sos1, 1)
            addsos(model, lp.sos2, 2)
            lp.solverModel = model
        except (xpress.ModelError, xpress.InterfaceError, xpress.SolverError) as err:
            self._reset(lp)
            raise PulpSolverError(str(err))

    def getAttribute(self, lp, which):
        """Get an arbitrary attribute for the model that was previously
        solved using actualSolve()."""
        return lp.solverModel.getAttrib(which)