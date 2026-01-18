import os
import sys
import ctypes
import subprocess
import warnings
from uuid import uuid4
from .core import sparse, ctypesArrayFill, PulpSolverError
from .core import clock, log
from .core import LpSolver, LpSolver_CMD
from ..constants import (
from ..constants import LpContinuous, LpBinary, LpInteger
from ..constants import LpConstraintEQ, LpConstraintLE, LpConstraintGE
from ..constants import LpMinimize, LpMaximize
class COPT_CMD(LpSolver_CMD):
    """
    The COPT command-line solver
    """
    name = 'COPT_CMD'

    def __init__(self, path=None, keepFiles=0, mip=True, msg=True, mip_start=False, warmStart=False, logfile=None, **params):
        """
        Initialize command-line solver
        """
        LpSolver_CMD.__init__(self, path, keepFiles, mip, msg, [])
        self.mipstart = warmStart
        self.logfile = logfile
        self.solverparams = params

    def defaultPath(self):
        """
        The default path of 'copt_cmd'
        """
        return self.executableExtension('copt_cmd')

    def available(self):
        """
        True if 'copt_cmd' is available
        """
        return self.executable(self.path)

    def actualSolve(self, lp):
        """
        Solve a well formulated LP problem

        This function borrowed implementation of CPLEX_CMD.actualSolve and
        GUROBI_CMD.actualSolve, with some modifications.
        """
        if not self.available():
            raise PulpSolverError("COPT_PULP: Failed to execute '{}'".format(self.path))
        if not self.keepFiles:
            uuid = uuid4().hex
            tmpLp = os.path.join(self.tmpDir, '{}-pulp.lp'.format(uuid))
            tmpSol = os.path.join(self.tmpDir, '{}-pulp.sol'.format(uuid))
            tmpMst = os.path.join(self.tmpDir, '{}-pulp.mst'.format(uuid))
        else:
            tmpName = lp.name
            tmpName = tmpName.replace(' ', '_')
            tmpLp = tmpName + '-pulp.lp'
            tmpSol = tmpName + '-pulp.sol'
            tmpMst = tmpName + '-pulp.mst'
        lpvars = lp.writeLP(tmpLp, writeSOS=1)
        solvecmds = self.path
        solvecmds += ' -c '
        solvecmds += '"read ' + tmpLp + ';'
        if lp.isMIP() and self.mipstart:
            self.writemst(tmpMst, lpvars)
            solvecmds += 'read ' + tmpMst + ';'
        if self.logfile is not None:
            solvecmds += 'set logfile {};'.format(self.logfile)
        if self.solverparams is not None:
            for parname, parval in self.solverparams.items():
                solvecmds += 'set {0} {1};'.format(parname, parval)
        if lp.isMIP() and (not self.mip):
            solvecmds += 'optimizelp;'
        else:
            solvecmds += 'optimize;'
        solvecmds += 'write ' + tmpSol + ';'
        solvecmds += 'exit"'
        try:
            os.remove(tmpSol)
        except:
            pass
        if self.msg:
            msgpipe = None
        else:
            msgpipe = open(os.devnull, 'w')
        rc = subprocess.call(solvecmds, shell=True, stdout=msgpipe, stderr=msgpipe)
        if msgpipe is not None:
            msgpipe.close()
        if rc != 0:
            raise PulpSolverError("COPT_PULP: Failed to execute '{}'".format(self.path))
        if not os.path.exists(tmpSol):
            status = LpStatusNotSolved
        else:
            status, values = self.readsol(tmpSol)
        if not self.keepFiles:
            for oldfile in [tmpLp, tmpSol, tmpMst]:
                try:
                    os.remove(oldfile)
                except:
                    pass
        if status == LpStatusOptimal:
            lp.assignVarsVals(values)
        lp.status = status
        return status

    def readsol(self, filename):
        """
        Read COPT solution file
        """
        with open(filename) as solfile:
            try:
                next(solfile)
            except StopIteration:
                warnings.warn('COPT_PULP: No solution was returned')
                return (LpStatusNotSolved, {})
            status = LpStatusOptimal
            values = {}
            for line in solfile:
                if line[0] != '#':
                    varname, varval = line.split()
                    values[varname] = float(varval)
        return (status, values)

    def writemst(self, filename, lpvars):
        """
        Write COPT MIP start file
        """
        mstvals = [(v.name, v.value()) for v in lpvars if v.value() is not None]
        mstline = []
        for varname, varval in mstvals:
            mstline.append('{0} {1}'.format(varname, varval))
        with open(filename, 'w') as mstfile:
            mstfile.write('\n'.join(mstline))
        return True