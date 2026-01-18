from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError, clock
from .core import glpk_path, operating_system, log
import os
from .. import constants
class GLPK_CMD(LpSolver_CMD):
    """The GLPK LP solver"""
    name = 'GLPK_CMD'

    def __init__(self, path=None, keepFiles=False, mip=True, msg=True, options=None, timeLimit=None):
        """
        :param bool mip: if False, assume LP even if integer variables
        :param bool msg: if False, no log is shown
        :param float timeLimit: maximum time for solver (in seconds)
        :param list options: list of additional options to pass to solver
        :param bool keepFiles: if True, files are saved in the current directory and not deleted after solving
        :param str path: path to the solver binary
        """
        LpSolver_CMD.__init__(self, mip=mip, msg=msg, timeLimit=timeLimit, options=options, path=path, keepFiles=keepFiles)

    def defaultPath(self):
        return self.executableExtension(glpk_path)

    def available(self):
        """True if the solver is available"""
        return self.executable(self.path)

    def actualSolve(self, lp):
        """Solve a well formulated lp problem"""
        if not self.executable(self.path):
            raise PulpSolverError('PuLP: cannot execute ' + self.path)
        tmpLp, tmpSol = self.create_tmp_files(lp.name, 'lp', 'sol')
        lp.writeLP(tmpLp, writeSOS=0)
        proc = ['glpsol', '--cpxlp', tmpLp, '-o', tmpSol]
        if self.timeLimit:
            proc.extend(['--tmlim', str(self.timeLimit)])
        if not self.mip:
            proc.append('--nomip')
        proc.extend(self.options)
        self.solution_time = clock()
        if not self.msg:
            proc[0] = self.path
            pipe = open(os.devnull, 'w')
            if operating_system == 'win':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                rc = subprocess.call(proc, stdout=pipe, stderr=pipe, startupinfo=startupinfo)
            else:
                rc = subprocess.call(proc, stdout=pipe, stderr=pipe)
            if rc:
                raise PulpSolverError('PuLP: Error while trying to execute ' + self.path)
            pipe.close()
        else:
            if os.name != 'nt':
                rc = os.spawnvp(os.P_WAIT, self.path, proc)
            else:
                rc = os.spawnv(os.P_WAIT, self.executable(self.path), proc)
            if rc == 127:
                raise PulpSolverError('PuLP: Error while trying to execute ' + self.path)
        self.solution_time += clock()
        if not os.path.exists(tmpSol):
            raise PulpSolverError('PuLP: Error while executing ' + self.path)
        status, values = self.readsol(tmpSol)
        lp.assignVarsVals(values)
        lp.assignStatus(status)
        self.delete_tmp_files(tmpLp, tmpSol)
        return status

    def readsol(self, filename):
        """Read a GLPK solution file"""
        with open(filename) as f:
            f.readline()
            rows = int(f.readline().split()[1])
            cols = int(f.readline().split()[1])
            f.readline()
            statusString = f.readline()[12:-1]
            glpkStatus = {'INTEGER OPTIMAL': constants.LpStatusOptimal, 'INTEGER NON-OPTIMAL': constants.LpStatusOptimal, 'OPTIMAL': constants.LpStatusOptimal, 'INFEASIBLE (FINAL)': constants.LpStatusInfeasible, 'INTEGER UNDEFINED': constants.LpStatusUndefined, 'UNBOUNDED': constants.LpStatusUnbounded, 'UNDEFINED': constants.LpStatusUndefined, 'INTEGER EMPTY': constants.LpStatusInfeasible}
            if statusString not in glpkStatus:
                raise PulpSolverError('Unknown status returned by GLPK')
            status = glpkStatus[statusString]
            isInteger = statusString in ['INTEGER NON-OPTIMAL', 'INTEGER OPTIMAL', 'INTEGER UNDEFINED', 'INTEGER EMPTY']
            values = {}
            for i in range(4):
                f.readline()
            for i in range(rows):
                line = f.readline().split()
                if len(line) == 2:
                    f.readline()
            for i in range(3):
                f.readline()
            for i in range(cols):
                line = f.readline().split()
                name = line[1]
                if len(line) == 2:
                    line = [0, 0] + f.readline().split()
                if isInteger:
                    if line[2] == '*':
                        value = int(float(line[3]))
                    else:
                        value = float(line[2])
                else:
                    value = float(line[3])
                values[name] = value
        return (status, values)