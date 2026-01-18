from math import inf
from typing import List
from .core import LpSolver, LpSolver_CMD, subprocess, PulpSolverError
import os, sys
from .. import constants
class HiGHS_CMD(LpSolver_CMD):
    """The HiGHS_CMD solver"""
    name: str = 'HiGHS_CMD'
    SOLUTION_STYLE: int = 0

    def __init__(self, path=None, keepFiles=False, mip=True, msg=True, options=None, timeLimit=None, gapRel=None, gapAbs=None, threads=None, logPath=None, warmStart=False):
        """
        :param bool mip: if False, assume LP even if integer variables
        :param bool msg: if False, no log is shown
        :param float timeLimit: maximum time for solver (in seconds)
        :param float gapRel: relative gap tolerance for the solver to stop (in fraction)
        :param float gapAbs: absolute gap tolerance for the solver to stop
        :param list[str] options: list of additional options to pass to solver
        :param bool keepFiles: if True, files are saved in the current directory and not deleted after solving
        :param str path: path to the solver binary (you can get binaries for your platform from https://github.com/JuliaBinaryWrappers/HiGHS_jll.jl/releases, or else compile from source - https://highs.dev)
        :param int threads: sets the maximum number of threads
        :param str logPath: path to the log file
        :param bool warmStart: if True, the solver will use the current value of variables as a start
        """
        LpSolver_CMD.__init__(self, mip=mip, msg=msg, timeLimit=timeLimit, gapRel=gapRel, gapAbs=gapAbs, options=options, path=path, keepFiles=keepFiles, threads=threads, logPath=logPath, warmStart=warmStart)

    def defaultPath(self):
        return self.executableExtension('highs')

    def available(self):
        """True if the solver is available"""
        return self.executable(self.path)

    def actualSolve(self, lp):
        """Solve a well formulated lp problem"""
        if not self.executable(self.path):
            raise PulpSolverError('PuLP: cannot execute ' + self.path)
        lp.checkDuplicateVars()
        tmpMps, tmpSol, tmpOptions, tmpLog, tmpMst = self.create_tmp_files(lp.name, 'mps', 'sol', 'HiGHS', 'HiGHS_log', 'mst')
        lp.writeMPS(tmpMps, with_objsense=True)
        file_options: List[str] = []
        file_options.append(f'solution_file={tmpSol}')
        file_options.append('write_solution_to_file=true')
        file_options.append(f'write_solution_style={HiGHS_CMD.SOLUTION_STYLE}')
        if not self.msg:
            file_options.append('log_to_console=false')
        if 'threads' in self.optionsDict:
            file_options.append(f'threads={self.optionsDict['threads']}')
        if 'gapRel' in self.optionsDict:
            file_options.append(f'mip_rel_gap={self.optionsDict['gapRel']}')
        if 'gapAbs' in self.optionsDict:
            file_options.append(f'mip_abs_gap={self.optionsDict['gapAbs']}')
        if 'logPath' in self.optionsDict:
            highs_log_file = self.optionsDict['logPath']
        else:
            highs_log_file = tmpLog
        file_options.append(f'log_file={highs_log_file}')
        command: List[str] = []
        command.append(self.path)
        command.append(tmpMps)
        command.append(f'--options_file={tmpOptions}')
        if self.timeLimit is not None:
            command.append(f'--time_limit={self.timeLimit}')
        if not self.mip:
            command.append('--solver=simplex')
        if 'threads' in self.optionsDict:
            command.append('--parallel=on')
        if self.optionsDict.get('warmStart', False):
            self.writesol(tmpMst, lp)
            command.append(f'--read_solution_file={tmpMst}')
        options = iter(self.options)
        for option in options:
            if '=' not in option:
                option += f'={next(options)}'
            if option.startswith('-'):
                command.append(option)
            else:
                file_options.append(option)
        with open(tmpOptions, 'w') as options_file:
            options_file.write('\n'.join(file_options))
        process = subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr)
        if process.returncode == -1:
            raise PulpSolverError('Error while executing HiGHS')
        with open(highs_log_file, 'r') as log_file:
            lines = log_file.readlines()
        lines = [line.strip().split() for line in lines]
        model_line = [line for line in lines if line[:2] == ['Model', 'status']]
        if len(model_line) > 0:
            model_status = ' '.join(model_line[0][3:])
        else:
            model_line = [line for line in lines if 'Status' in line][0]
            model_status = ' '.join(model_line[1:])
        sol_line = [line for line in lines if line[:2] == ['Solution', 'status']]
        sol_line = sol_line[0] if len(sol_line) > 0 else ['Not solved']
        sol_status = sol_line[-1]
        if model_status.lower() == 'optimal':
            status, status_sol = (constants.LpStatusOptimal, constants.LpSolutionOptimal)
        elif sol_status.lower() == 'feasible':
            status, status_sol = (constants.LpStatusOptimal, constants.LpSolutionIntegerFeasible)
        elif model_status.lower() == 'infeasible':
            status, status_sol = (constants.LpStatusInfeasible, constants.LpSolutionInfeasible)
        elif model_status.lower() == 'unbounded':
            status, status_sol = (constants.LpStatusUnbounded, constants.LpSolutionUnbounded)
        else:
            status, status_sol = (constants.LpStatusNotSolved, constants.LpSolutionNoSolutionFound)
        if not os.path.exists(tmpSol) or os.stat(tmpSol).st_size == 0:
            status_sol = constants.LpSolutionNoSolutionFound
            values = None
        elif status_sol == constants.LpSolutionNoSolutionFound:
            values = None
        else:
            values = self.readsol(tmpSol)
        self.delete_tmp_files(tmpMps, tmpSol, tmpOptions, tmpLog, tmpMst)
        lp.assignStatus(status, status_sol)
        if status == constants.LpStatusOptimal:
            lp.assignVarsVals(values)
        return status

    def writesol(self, filename, lp):
        """Writes a HiGHS solution file"""
        variable_rows = []
        for var in lp.variables():
            variable_rows.append(f'{var.name} {var.varValue or 0}')
        all_rows = ['Model status', 'None', '', '# Primal solution values', 'Feasible', '', f'# Columns {len(variable_rows)}']
        all_rows.extend(variable_rows)
        with open(filename, 'w') as file:
            file.write('\n'.join(all_rows))

    def readsol(self, filename):
        """Read a HiGHS solution file"""
        with open(filename) as file:
            lines = file.readlines()
        begin, end = (None, None)
        for index, line in enumerate(lines):
            if begin is None and line.startswith('# Columns'):
                begin = index + 1
            if end is None and line.startswith('# Rows'):
                end = index
        if begin is None or end is None:
            raise PulpSolverError('Cannot read HiGHS solver output')
        values = {}
        for line in lines[begin:end]:
            name, value = line.split()
            values[name] = float(value)
        return values