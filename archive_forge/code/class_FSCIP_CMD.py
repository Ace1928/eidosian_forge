import operator
import os
import sys
import warnings
from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError
from .core import scip_path, fscip_path
from .. import constants
from typing import Dict, List, Optional, Tuple
class FSCIP_CMD(LpSolver_CMD):
    """The multi-threaded FiberSCIP version of the SCIP optimization solver"""
    name = 'FSCIP_CMD'

    def __init__(self, path=None, mip=True, keepFiles=False, msg=True, options=None, timeLimit=None, gapRel=None, gapAbs=None, maxNodes=None, threads=None, logPath=None):
        """
        :param bool msg: if False, no log is shown
        :param bool mip: if False, assume LP even if integer variables
        :param list options: list of additional options to pass to solver
        :param bool keepFiles: if True, files are saved in the current directory and not deleted after solving
        :param str path: path to the solver binary
        :param float timeLimit: maximum time for solver (in seconds)
        :param float gapRel: relative gap tolerance for the solver to stop (in fraction)
        :param float gapAbs: absolute gap tolerance for the solver to stop
        :param int maxNodes: max number of nodes during branching. Stops the solving when reached.
        :param int threads: sets the maximum number of threads
        :param str logPath: path to the log file
        """
        LpSolver_CMD.__init__(self, mip=mip, msg=msg, options=options, path=path, keepFiles=keepFiles, timeLimit=timeLimit, gapRel=gapRel, gapAbs=gapAbs, maxNodes=maxNodes, threads=threads, logPath=logPath)
    FSCIP_STATUSES = {'No Solution': constants.LpStatusNotSolved, 'Final Solution': constants.LpStatusOptimal}
    NO_SOLUTION_STATUSES = {constants.LpStatusInfeasible, constants.LpStatusUnbounded, constants.LpStatusNotSolved}

    def defaultPath(self):
        return self.executableExtension(fscip_path)

    def available(self):
        """True if the solver is available"""
        return self.executable(self.path)

    def actualSolve(self, lp):
        """Solve a well formulated lp problem"""
        if not self.executable(self.path):
            raise PulpSolverError('PuLP: cannot execute ' + self.path)
        tmpLp, tmpSol, tmpOptions, tmpParams = self.create_tmp_files(lp.name, 'lp', 'sol', 'set', 'prm')
        lp.writeLP(tmpLp)
        file_options: List[str] = []
        if self.timeLimit is not None:
            file_options.append(f'limits/time={self.timeLimit}')
        if 'gapRel' in self.optionsDict:
            file_options.append(f'limits/gap={self.optionsDict['gapRel']}')
        if 'gapAbs' in self.optionsDict:
            file_options.append(f'limits/absgap={self.optionsDict['gapAbs']}')
        if 'maxNodes' in self.optionsDict:
            file_options.append(f'limits/nodes={self.optionsDict['maxNodes']}')
        if not self.mip:
            warnings.warn(f'{self.name} does not allow a problem to be relaxed')
        file_parameters: List[str] = []
        file_parameters.append('NoPreprocessingInLC = TRUE')
        command: List[str] = []
        command.append(self.path)
        command.append(tmpParams)
        command.append(tmpLp)
        command.extend(['-s', tmpOptions])
        command.extend(['-fsol', tmpSol])
        if not self.msg:
            command.append('-q')
        if 'logPath' in self.optionsDict:
            command.extend(['-l', self.optionsDict['logPath']])
        if 'threads' in self.optionsDict:
            command.extend(['-sth', f'{self.optionsDict['threads']}'])
        options = iter(self.options)
        for option in options:
            if option.startswith('-'):
                argument = next(options)
                command.extend([option, argument])
            else:
                is_file_options = '/' in option
                if '=' not in option:
                    argument = next(options)
                    option += f'={argument}'
                if is_file_options:
                    file_options.append(option)
                else:
                    file_parameters.append(option)
        self.silent_remove(tmpSol)
        with open(tmpOptions, 'w') as options_file:
            options_file.write('\n'.join(file_options))
        with open(tmpParams, 'w') as parameters_file:
            parameters_file.write('\n'.join(file_parameters))
        subprocess.check_call(command, stdout=sys.stdout if self.msg else subprocess.DEVNULL, stderr=sys.stderr if self.msg else subprocess.DEVNULL)
        if not os.path.exists(tmpSol):
            raise PulpSolverError('PuLP: Error while executing ' + self.path)
        status, values = self.readsol(tmpSol)
        finalVals = {}
        for v in lp.variables():
            finalVals[v.name] = values.get(v.name, 0.0)
        lp.assignVarsVals(finalVals)
        lp.assignStatus(status)
        self.delete_tmp_files(tmpLp, tmpSol, tmpOptions, tmpParams)
        return status

    @staticmethod
    def parse_status(string: str) -> Optional[int]:
        for fscip_status, pulp_status in FSCIP_CMD.FSCIP_STATUSES.items():
            if fscip_status in string:
                return pulp_status
        return None

    @staticmethod
    def parse_objective(string: str) -> Optional[float]:
        fields = string.split(':')
        if len(fields) != 2:
            return None
        label, objective = fields
        if label != 'objective value':
            return None
        objective = objective.strip()
        try:
            objective = float(objective)
        except ValueError:
            return None
        return objective

    @staticmethod
    def parse_variable(string: str) -> Optional[Tuple[str, float]]:
        fields = string.split()
        if len(fields) < 2:
            return None
        name, value = fields[:2]
        try:
            value = float(value)
        except ValueError:
            return None
        return (name, value)

    @staticmethod
    def readsol(filename):
        """Read a FSCIP solution file"""
        with open(filename) as file:
            status_line = file.readline()
            status = FSCIP_CMD.parse_status(status_line)
            if status is None:
                raise PulpSolverError(f"Can't get FSCIP solver status: {status_line!r}")
            if status in FSCIP_CMD.NO_SOLUTION_STATUSES:
                return (status, {})
            objective_line = file.readline()
            objective = FSCIP_CMD.parse_objective(objective_line)
            if objective is None:
                raise PulpSolverError(f"Can't get FSCIP solver objective: {objective_line!r}")
            variables: Dict[str, float] = {}
            for variable_line in file:
                variable = FSCIP_CMD.parse_variable(variable_line)
                if variable is None:
                    raise PulpSolverError(f"Can't read FSCIP solver output: {variable_line!r}")
                name, value = variable
                variables[name] = value
            return (status, variables)