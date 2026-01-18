from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError, clock, log
from .core import cbc_path, pulp_cbc_path, coinMP_path, devnull, operating_system
import os
from .. import constants
from tempfile import mktemp
import ctypes
import warnings
class COIN_CMD(LpSolver_CMD):
    """The COIN CLP/CBC LP solver
    now only uses cbc
    """
    name = 'COIN_CMD'

    def defaultPath(self):
        return self.executableExtension(cbc_path)

    def __init__(self, mip=True, msg=True, timeLimit=None, gapRel=None, gapAbs=None, presolve=None, cuts=None, strong=None, options=None, warmStart=False, keepFiles=False, path=None, threads=None, logPath=None, timeMode='elapsed', maxNodes=None):
        """
        :param bool mip: if False, assume LP even if integer variables
        :param bool msg: if False, no log is shown
        :param float timeLimit: maximum time for solver (in seconds)
        :param float gapRel: relative gap tolerance for the solver to stop (in fraction)
        :param float gapAbs: absolute gap tolerance for the solver to stop
        :param int threads: sets the maximum number of threads
        :param list options: list of additional options to pass to solver
        :param bool warmStart: if True, the solver will use the current value of variables as a start
        :param bool keepFiles: if True, files are saved in the current directory and not deleted after solving
        :param str path: path to the solver binary
        :param str logPath: path to the log file
        :param bool presolve: if True, adds presolve on
        :param bool cuts: if True, adds gomory on knapsack on probing on
        :param bool strong: if True, adds strong
        :param str timeMode: "elapsed": count wall-time to timeLimit; "cpu": count cpu-time
        :param int maxNodes: max number of nodes during branching. Stops the solving when reached.
        """
        LpSolver_CMD.__init__(self, gapRel=gapRel, mip=mip, msg=msg, timeLimit=timeLimit, presolve=presolve, cuts=cuts, strong=strong, options=options, warmStart=warmStart, path=path, keepFiles=keepFiles, threads=threads, gapAbs=gapAbs, logPath=logPath, timeMode=timeMode, maxNodes=maxNodes)

    def copy(self):
        """Make a copy of self"""
        aCopy = LpSolver_CMD.copy(self)
        aCopy.optionsDict = self.optionsDict
        return aCopy

    def actualSolve(self, lp, **kwargs):
        """Solve a well formulated lp problem"""
        return self.solve_CBC(lp, **kwargs)

    def available(self):
        """True if the solver is available"""
        return self.executable(self.path)

    def solve_CBC(self, lp, use_mps=True):
        """Solve a MIP problem using CBC"""
        if not self.executable(self.path):
            raise PulpSolverError(f'Pulp: cannot execute {self.path} cwd: {os.getcwd()}')
        tmpLp, tmpMps, tmpSol, tmpMst = self.create_tmp_files(lp.name, 'lp', 'mps', 'sol', 'mst')
        if use_mps:
            vs, variablesNames, constraintsNames, objectiveName = lp.writeMPS(tmpMps, rename=1)
            cmds = ' ' + tmpMps + ' '
            if lp.sense == constants.LpMaximize:
                cmds += '-max '
        else:
            vs = lp.writeLP(tmpLp)
            variablesNames = {v.name: v.name for v in vs}
            constraintsNames = {c: c for c in lp.constraints}
            cmds = ' ' + tmpLp + ' '
        if self.optionsDict.get('warmStart', False):
            self.writesol(tmpMst, lp, vs, variablesNames, constraintsNames)
            cmds += f'-mips {tmpMst} '
        if self.timeLimit is not None:
            cmds += f'-sec {self.timeLimit} '
        options = self.options + self.getOptions()
        for option in options:
            cmds += '-' + option + ' '
        if self.mip:
            cmds += '-branch '
        else:
            cmds += '-initialSolve '
        cmds += '-printingOptions all '
        cmds += '-solution ' + tmpSol + ' '
        if self.msg:
            pipe = None
        else:
            pipe = open(os.devnull, 'w')
        logPath = self.optionsDict.get('logPath')
        if logPath:
            if self.msg:
                warnings.warn('`logPath` argument replaces `msg=1`. The output will be redirected to the log file.')
            pipe = open(self.optionsDict['logPath'], 'w')
        log.debug(self.path + cmds)
        args = []
        args.append(self.path)
        args.extend(cmds[1:].split())
        if not self.msg and operating_system == 'win':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            cbc = subprocess.Popen(args, stdout=pipe, stderr=pipe, stdin=devnull, startupinfo=startupinfo)
        else:
            cbc = subprocess.Popen(args, stdout=pipe, stderr=pipe, stdin=devnull)
        if cbc.wait() != 0:
            if pipe:
                pipe.close()
            raise PulpSolverError('Pulp: Error while trying to execute, use msg=True for more details' + self.path)
        if pipe:
            pipe.close()
        if not os.path.exists(tmpSol):
            raise PulpSolverError('Pulp: Error while executing ' + self.path)
        status, values, reducedCosts, shadowPrices, slacks, sol_status = self.readsol_MPS(tmpSol, lp, vs, variablesNames, constraintsNames)
        lp.assignVarsVals(values)
        lp.assignVarsDj(reducedCosts)
        lp.assignConsPi(shadowPrices)
        lp.assignConsSlack(slacks, activity=True)
        lp.assignStatus(status, sol_status)
        self.delete_tmp_files(tmpMps, tmpLp, tmpSol, tmpMst)
        return status

    def getOptions(self):
        params_eq = dict(gapRel='ratio {}', gapAbs='allow {}', threads='threads {}', presolve='presolve on', strong='strong {}', cuts='gomory on knapsack on probing on', timeMode='timeMode {}', maxNodes='maxNodes {}')
        return [v.format(self.optionsDict[k]) for k, v in params_eq.items() if self.optionsDict.get(k) is not None]

    def readsol_MPS(self, filename, lp, vs, variablesNames, constraintsNames, objectiveName=None):
        """
        Read a CBC solution file generated from an mps or lp file (possible different names)
        """
        values = {v.name: 0 for v in vs}
        reverseVn = {v: k for k, v in variablesNames.items()}
        reverseCn = {v: k for k, v in constraintsNames.items()}
        reducedCosts = {}
        shadowPrices = {}
        slacks = {}
        status, sol_status = self.get_status(filename)
        with open(filename) as f:
            for l in f:
                if len(l) <= 2:
                    break
                l = l.split()
                if l[0] == '**':
                    l = l[1:]
                vn = l[1]
                val = l[2]
                dj = l[3]
                if vn in reverseVn:
                    values[reverseVn[vn]] = float(val)
                    reducedCosts[reverseVn[vn]] = float(dj)
                if vn in reverseCn:
                    slacks[reverseCn[vn]] = float(val)
                    shadowPrices[reverseCn[vn]] = float(dj)
        return (status, values, reducedCosts, shadowPrices, slacks, sol_status)

    def writesol(self, filename, lp, vs, variablesNames, constraintsNames):
        """
        Writes a CBC solution file generated from an mps / lp file (possible different names)
        returns True on success
        """
        values = {v.name: v.value() if v.value() is not None else 0 for v in vs}
        value_lines = []
        value_lines += [(i, v, values[k], 0) for i, (k, v) in enumerate(variablesNames.items())]
        lines = ['Stopped on time - objective value 0\n']
        lines += ['{:>7} {} {:>15} {:>23}\n'.format(*tup) for tup in value_lines]
        with open(filename, 'w') as f:
            f.writelines(lines)
        return True

    def readsol_LP(self, filename, lp, vs):
        """
        Read a CBC solution file generated from an lp (good names)
        returns status, values, reducedCosts, shadowPrices, slacks, sol_status
        """
        variablesNames = {v.name: v.name for v in vs}
        constraintsNames = {c: c for c in lp.constraints}
        return self.readsol_MPS(filename, lp, vs, variablesNames, constraintsNames)

    def get_status(self, filename):
        cbcStatus = {'Optimal': constants.LpStatusOptimal, 'Infeasible': constants.LpStatusInfeasible, 'Integer': constants.LpStatusInfeasible, 'Unbounded': constants.LpStatusUnbounded, 'Stopped': constants.LpStatusNotSolved}
        cbcSolStatus = {'Optimal': constants.LpSolutionOptimal, 'Infeasible': constants.LpSolutionInfeasible, 'Unbounded': constants.LpSolutionUnbounded, 'Stopped': constants.LpSolutionNoSolutionFound}
        with open(filename) as f:
            statusstrs = f.readline().split()
        status = cbcStatus.get(statusstrs[0], constants.LpStatusUndefined)
        sol_status = cbcSolStatus.get(statusstrs[0], constants.LpSolutionNoSolutionFound)
        if status == constants.LpStatusNotSolved and len(statusstrs) >= 5:
            if statusstrs[4] == 'objective':
                status = constants.LpStatusOptimal
                sol_status = constants.LpSolutionIntegerFeasible
        return (status, sol_status)