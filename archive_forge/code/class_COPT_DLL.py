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
class COPT_DLL(LpSolver):
    """
    The COPT dynamic library solver
    """
    name = 'COPT_DLL'
    try:
        coptlib = COPT_DLL_loadlib()
    except Exception as e:
        err = e
        'The COPT dynamic library solver (DLL). Something went wrong!!!!'

        def available(self):
            """True if the solver is available"""
            return False

        def actualSolve(self, lp):
            """Solve a well formulated lp problem"""
            raise PulpSolverError(f'COPT_DLL: Not Available:\n{self.err}')
    else:
        CreateEnv = coptlib.COPT_CreateEnv
        DeleteEnv = coptlib.COPT_DeleteEnv
        CreateProb = coptlib.COPT_CreateProb
        DeleteProb = coptlib.COPT_DeleteProb
        LoadProb = coptlib.COPT_LoadProb
        AddCols = coptlib.COPT_AddCols
        WriteMps = coptlib.COPT_WriteMps
        WriteLp = coptlib.COPT_WriteLp
        WriteBin = coptlib.COPT_WriteBin
        WriteSol = coptlib.COPT_WriteSol
        WriteBasis = coptlib.COPT_WriteBasis
        WriteMst = coptlib.COPT_WriteMst
        WriteParam = coptlib.COPT_WriteParam
        AddMipStart = coptlib.COPT_AddMipStart
        SolveLp = coptlib.COPT_SolveLp
        Solve = coptlib.COPT_Solve
        GetSolution = coptlib.COPT_GetSolution
        GetLpSolution = coptlib.COPT_GetLpSolution
        GetIntParam = coptlib.COPT_GetIntParam
        SetIntParam = coptlib.COPT_SetIntParam
        GetDblParam = coptlib.COPT_GetDblParam
        SetDblParam = coptlib.COPT_SetDblParam
        GetIntAttr = coptlib.COPT_GetIntAttr
        GetDblAttr = coptlib.COPT_GetDblAttr
        SearchParamAttr = coptlib.COPT_SearchParamAttr
        SetLogFile = coptlib.COPT_SetLogFile

        def __init__(self, mip=True, msg=True, mip_start=False, warmStart=False, logfile=None, **params):
            """
            Initialize COPT solver
            """
            LpSolver.__init__(self, mip, msg)
            self.coptenv = None
            self.coptprob = None
            self.mipstart = warmStart
            self.create()
            if logfile is not None:
                rc = self.SetLogFile(self.coptprob, coptstr(logfile))
                if rc != 0:
                    raise PulpSolverError('COPT_PULP: Failed to set log file')
            if not self.msg:
                self.setParam('Logging', 0)
            for parname, parval in params.items():
                self.setParam(parname, parval)

        def available(self):
            """
            True if dynamic library is available
            """
            return True

        def actualSolve(self, lp):
            """
            Solve a well formulated LP/MIP problem

            This function borrowed implementation of CPLEX_DLL.actualSolve,
            with some modifications.
            """
            ncol, nrow, nnonz, objsen, objconst, colcost, colbeg, colcnt, colind, colval, coltype, collb, colub, rowsense, rowrhs, colname, rowname = self.extract(lp)
            rc = self.LoadProb(self.coptprob, ncol, nrow, objsen, objconst, colcost, colbeg, colcnt, colind, colval, coltype, collb, colub, rowsense, rowrhs, None, colname, rowname)
            if rc != 0:
                raise PulpSolverError('COPT_PULP: Failed to load problem')
            if lp.isMIP() and self.mip:
                if self.mipstart:
                    mstdict = {self.v2n[v]: v.value() for v in lp.variables() if v.value() is not None}
                    if mstdict:
                        mstkeys = ctypesArrayFill(list(mstdict.keys()), ctypes.c_int)
                        mstvals = ctypesArrayFill(list(mstdict.values()), ctypes.c_double)
                        rc = self.AddMipStart(self.coptprob, len(mstkeys), mstkeys, mstvals)
                        if rc != 0:
                            raise PulpSolverError('COPT_PULP: Failed to add MIP start information')
                rc = self.Solve(self.coptprob)
                if rc != 0:
                    raise PulpSolverError('COPT_PULP: Failed to solve the MIP problem')
            elif lp.isMIP() and (not self.mip):
                rc = self.SolveLp(self.coptprob)
                if rc != 0:
                    raise PulpSolverError('COPT_PULP: Failed to solve MIP as LP')
            else:
                rc = self.SolveLp(self.coptprob)
                if rc != 0:
                    raise PulpSolverError('COPT_PULP: Failed to solve the LP problem')
            status = self.getsolution(lp, ncol, nrow)
            for var in lp.variables():
                var.modified = False
            return status

        def extract(self, lp):
            """
            Extract data from PuLP lp structure

            This function borrowed implementation of LpSolver.getCplexStyleArrays,
            with some modifications.
            """
            cols = list(lp.variables())
            ncol = len(cols)
            nrow = len(lp.constraints)
            collb = (ctypes.c_double * ncol)()
            colub = (ctypes.c_double * ncol)()
            colcost = (ctypes.c_double * ncol)()
            coltype = (ctypes.c_char * ncol)()
            colname = (ctypes.c_char_p * ncol)()
            rowrhs = (ctypes.c_double * nrow)()
            rowsense = (ctypes.c_char * nrow)()
            rowname = (ctypes.c_char_p * nrow)()
            spmat = sparse.Matrix(list(range(nrow)), list(range(ncol)))
            objsen = coptobjsen[lp.sense]
            objconst = ctypes.c_double(0.0)
            self.v2n = dict(((cols[i], i) for i in range(ncol)))
            self.vname2n = dict(((cols[i].name, i) for i in range(ncol)))
            self.n2v = dict(((i, cols[i]) for i in range(ncol)))
            self.c2n = {}
            self.n2c = {}
            self.addedVars = ncol
            self.addedRows = nrow
            for col, val in lp.objective.items():
                colcost[self.v2n[col]] = val
            for col in lp.variables():
                colname[self.v2n[col]] = coptstr(col.name)
                if col.lowBound is not None:
                    collb[self.v2n[col]] = col.lowBound
                else:
                    collb[self.v2n[col]] = -1e+30
                if col.upBound is not None:
                    colub[self.v2n[col]] = col.upBound
                else:
                    colub[self.v2n[col]] = 1e+30
            if lp.isMIP():
                for var in lp.variables():
                    coltype[self.v2n[var]] = coptctype[var.cat]
            else:
                coltype = None
            idx = 0
            for row in lp.constraints:
                rowrhs[idx] = -lp.constraints[row].constant
                rowsense[idx] = coptrsense[lp.constraints[row].sense]
                rowname[idx] = coptstr(row)
                self.c2n[row] = idx
                self.n2c[idx] = row
                idx += 1
            for col, row, coeff in lp.coefficients():
                spmat.add(self.c2n[row], self.vname2n[col], coeff)
            nnonz, _colbeg, _colcnt, _colind, _colval = spmat.col_based_arrays()
            colbeg = ctypesArrayFill(_colbeg, ctypes.c_int)
            colcnt = ctypesArrayFill(_colcnt, ctypes.c_int)
            colind = ctypesArrayFill(_colind, ctypes.c_int)
            colval = ctypesArrayFill(_colval, ctypes.c_double)
            return (ncol, nrow, nnonz, objsen, objconst, colcost, colbeg, colcnt, colind, colval, coltype, collb, colub, rowsense, rowrhs, colname, rowname)

        def create(self):
            """
            Create COPT environment and problem

            This function borrowed implementation of CPLEX_DLL.grabLicense,
            with some modifications.
            """
            self.delete()
            self.coptenv = ctypes.c_void_p()
            self.coptprob = ctypes.c_void_p()
            rc = self.CreateEnv(byref(self.coptenv))
            if rc != 0:
                raise PulpSolverError('COPT_PULP: Failed to create environment')
            rc = self.CreateProb(self.coptenv, byref(self.coptprob))
            if rc != 0:
                raise PulpSolverError('COPT_PULP: Failed to create problem')

        def __del__(self):
            """
            Destructor of COPT_DLL class
            """
            self.delete()

        def delete(self):
            """
            Release COPT problem and environment

            This function borrowed implementation of CPLEX_DLL.releaseLicense,
            with some modifications.
            """
            if self.coptenv is not None and self.coptprob is not None:
                rc = self.DeleteProb(byref(self.coptprob))
                if rc != 0:
                    raise PulpSolverError('COPT_PULP: Failed to delete problem')
                rc = self.DeleteEnv(byref(self.coptenv))
                if rc != 0:
                    raise PulpSolverError('COPT_PULP: Failed to delete environment')
                self.coptenv = None
                self.coptprob = None

        def getsolution(self, lp, ncols, nrows):
            """Get problem solution

            This function borrowed implementation of CPLEX_DLL.findSolutionValues,
            with some modifications.
            """
            status = ctypes.c_int()
            x = (ctypes.c_double * ncols)()
            dj = (ctypes.c_double * ncols)()
            pi = (ctypes.c_double * nrows)()
            slack = (ctypes.c_double * nrows)()
            var_x = {}
            var_dj = {}
            con_pi = {}
            con_slack = {}
            if lp.isMIP() and self.mip:
                hasmipsol = ctypes.c_int()
                rc = self.GetIntAttr(self.coptprob, coptstr('MipStatus'), byref(status))
                if rc != 0:
                    raise PulpSolverError('COPT_PULP: Failed to get MIP status')
                rc = self.GetIntAttr(self.coptprob, coptstr('HasMipSol'), byref(hasmipsol))
                if rc != 0:
                    raise PulpSolverError('COPT_PULP: Failed to check if MIP solution exists')
                if status.value == 1 or hasmipsol.value == 1:
                    rc = self.GetSolution(self.coptprob, byref(x))
                    if rc != 0:
                        raise PulpSolverError('COPT_PULP: Failed to get MIP solution')
                    for i in range(ncols):
                        var_x[self.n2v[i].name] = x[i]
                lp.assignVarsVals(var_x)
            else:
                rc = self.GetIntAttr(self.coptprob, coptstr('LpStatus'), byref(status))
                if rc != 0:
                    raise PulpSolverError('COPT_PULP: Failed to get LP status')
                if status.value == 1:
                    rc = self.GetLpSolution(self.coptprob, byref(x), byref(slack), byref(pi), byref(dj))
                    if rc != 0:
                        raise PulpSolverError('COPT_PULP: Failed to get LP solution')
                    for i in range(ncols):
                        var_x[self.n2v[i].name] = x[i]
                        var_dj[self.n2v[i].name] = dj[i]
                    for i in range(nrows):
                        con_pi[self.n2c[i]] = pi[i]
                        con_slack[self.n2c[i]] = slack[i]
                lp.assignVarsVals(var_x)
                lp.assignVarsDj(var_dj)
                lp.assignConsPi(con_pi)
                lp.assignConsSlack(con_slack)
            lp.resolveOK = True
            for var in lp.variables():
                var.isModified = False
            lp.status = coptlpstat.get(status.value, LpStatusUndefined)
            return lp.status

        def write(self, filename):
            """
            Write problem, basis, parameter or solution to file
            """
            file_path = coptstr(filename)
            file_name, file_ext = os.path.splitext(file_path)
            if not file_ext:
                raise PulpSolverError('COPT_PULP: Failed to determine output file type')
            elif file_ext == coptstr('.mps'):
                rc = self.WriteMps(self.coptprob, file_path)
            elif file_ext == coptstr('.lp'):
                rc = self.WriteLp(self.coptprob, file_path)
            elif file_ext == coptstr('.bin'):
                rc = self.WriteBin(self.coptprob, file_path)
            elif file_ext == coptstr('.sol'):
                rc = self.WriteSol(self.coptprob, file_path)
            elif file_ext == coptstr('.bas'):
                rc = self.WriteBasis(self.coptprob, file_path)
            elif file_ext == coptstr('.mst'):
                rc = self.WriteMst(self.coptprob, file_path)
            elif file_ext == coptstr('.par'):
                rc = self.WriteParam(self.coptprob, file_path)
            else:
                raise PulpSolverError('COPT_PULP: Unsupported file type')
            if rc != 0:
                raise PulpSolverError("COPT_PULP: Failed to write file '{}'".format(filename))

        def setParam(self, name, val):
            """
            Set parameter to COPT problem
            """
            par_type = ctypes.c_int()
            par_name = coptstr(name)
            rc = self.SearchParamAttr(self.coptprob, par_name, byref(par_type))
            if rc != 0:
                raise PulpSolverError("COPT_PULP: Failed to check type for '{}'".format(par_name))
            if par_type.value == 0:
                rc = self.SetDblParam(self.coptprob, par_name, ctypes.c_double(val))
                if rc != 0:
                    raise PulpSolverError("COPT_PULP: Failed to set double parameter '{}'".format(par_name))
            elif par_type.value == 1:
                rc = self.SetIntParam(self.coptprob, par_name, ctypes.c_int(val))
                if rc != 0:
                    raise PulpSolverError("COPT_PULP: Failed to set integer parameter '{}'".format(par_name))
            else:
                raise PulpSolverError("COPT_PULP: Invalid parameter '{}'".format(par_name))

        def getParam(self, name):
            """
            Get current value of parameter
            """
            par_dblval = ctypes.c_double()
            par_intval = ctypes.c_int()
            par_type = ctypes.c_int()
            par_name = coptstr(name)
            rc = self.SearchParamAttr(self.coptprob, par_name, byref(par_type))
            if rc != 0:
                raise PulpSolverError("COPT_PULP: Failed to check type for '{}'".format(par_name))
            if par_type.value == 0:
                rc = self.GetDblParam(self.coptprob, par_name, byref(par_dblval))
                if rc != 0:
                    raise PulpSolverError("COPT_PULP: Failed to get double parameter '{}'".format(par_name))
                else:
                    retval = par_dblval.value
            elif par_type.value == 1:
                rc = self.GetIntParam(self.coptprob, par_name, byref(par_intval))
                if rc != 0:
                    raise PulpSolverError("COPT_PULP: Failed to get integer parameter '{}'".format(par_name))
                else:
                    retval = par_intval.value
            else:
                raise PulpSolverError("COPT_PULP: Invalid parameter '{}'".format(par_name))
            return retval

        def getAttr(self, name):
            """
            Get attribute of the problem
            """
            attr_dblval = ctypes.c_double()
            attr_intval = ctypes.c_int()
            attr_type = ctypes.c_int()
            attr_name = coptstr(name)
            rc = self.SearchParamAttr(self.coptprob, attr_name, byref(attr_type))
            if rc != 0:
                raise PulpSolverError("COPT_PULP: Failed to check type for '{}'".format(attr_name))
            if attr_type.value == 2:
                rc = self.GetDblAttr(self.coptprob, attr_name, byref(attr_dblval))
                if rc != 0:
                    raise PulpSolverError("COPT_PULP: Failed to get double attribute '{}'".format(attr_name))
                else:
                    retval = attr_dblval.value
            elif attr_type.value == 3:
                rc = self.GetIntAttr(self.coptprob, attr_name, byref(attr_intval))
                if rc != 0:
                    raise PulpSolverError("COPT_PULP: Failed to get integer attribute '{}'".format(attr_name))
                else:
                    retval = attr_intval.value
            else:
                raise PulpSolverError("COPT_PULP: Invalid attribute '{}'".format(attr_name))
            return retval