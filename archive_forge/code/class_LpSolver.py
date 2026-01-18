import os
import platform
import shutil
import sys
import ctypes
from time import monotonic as clock
import configparser
from typing import Union
from .. import sparse
from .. import constants as const
import logging
import subprocess
from uuid import uuid4
class LpSolver:
    """A generic LP Solver"""
    name = 'LpSolver'

    def __init__(self, mip=True, msg=True, options=None, timeLimit=None, *args, **kwargs):
        """
        :param bool mip: if False, assume LP even if integer variables
        :param bool msg: if False, no log is shown
        :param list options:
        :param float timeLimit: maximum time for solver (in seconds)
        :param args:
        :param kwargs: optional named options to pass to each solver,
                        e.g. gapRel=0.1, gapAbs=10, logPath="",
        """
        if options is None:
            options = []
        self.mip = mip
        self.msg = msg
        self.options = options
        self.timeLimit = timeLimit
        self.optionsDict = {k: v for k, v in kwargs.items() if v is not None}

    def available(self):
        """True if the solver is available"""
        raise NotImplementedError

    def actualSolve(self, lp):
        """Solve a well formulated lp problem"""
        raise NotImplementedError

    def actualResolve(self, lp, **kwargs):
        """
        uses existing problem information and solves the problem
        If it is not implemented in the solver
        just solve again
        """
        self.actualSolve(lp, **kwargs)

    def copy(self):
        """Make a copy of self"""
        aCopy = self.__class__()
        aCopy.mip = self.mip
        aCopy.msg = self.msg
        aCopy.options = self.options
        return aCopy

    def solve(self, lp):
        """Solve the problem lp"""
        return lp.solve(self)

    def getCplexStyleArrays(self, lp, senseDict=None, LpVarCategories=None, LpObjSenses=None, infBound=1e+20):
        """returns the arrays suitable to pass to a cdll Cplex
        or other solvers that are similar

        Copyright (c) Stuart Mitchell 2007
        """
        if senseDict is None:
            senseDict = {const.LpConstraintEQ: 'E', const.LpConstraintLE: 'L', const.LpConstraintGE: 'G'}
        if LpVarCategories is None:
            LpVarCategories = {const.LpContinuous: 'C', const.LpInteger: 'I'}
        if LpObjSenses is None:
            LpObjSenses = {const.LpMaximize: -1, const.LpMinimize: 1}
        import ctypes
        rangeCount = 0
        variables = list(lp.variables())
        numVars = len(variables)
        self.v2n = {variables[i]: i for i in range(numVars)}
        self.vname2n = {variables[i].name: i for i in range(numVars)}
        self.n2v = {i: variables[i] for i in range(numVars)}
        objSense = LpObjSenses[lp.sense]
        NumVarDoubleArray = ctypes.c_double * numVars
        objectCoeffs = NumVarDoubleArray()
        for v, val in lp.objective.items():
            objectCoeffs[self.v2n[v]] = val
        objectConst = ctypes.c_double(0.0)
        NumVarStrArray = ctypes.c_char_p * numVars
        colNames = NumVarStrArray()
        lowerBounds = NumVarDoubleArray()
        upperBounds = NumVarDoubleArray()
        initValues = NumVarDoubleArray()
        for v in lp.variables():
            colNames[self.v2n[v]] = to_string(v.name)
            initValues[self.v2n[v]] = 0.0
            if v.lowBound != None:
                lowerBounds[self.v2n[v]] = v.lowBound
            else:
                lowerBounds[self.v2n[v]] = -infBound
            if v.upBound != None:
                upperBounds[self.v2n[v]] = v.upBound
            else:
                upperBounds[self.v2n[v]] = infBound
        numRows = len(lp.constraints)
        NumRowDoubleArray = ctypes.c_double * numRows
        NumRowStrArray = ctypes.c_char_p * numRows
        NumRowCharArray = ctypes.c_char * numRows
        rhsValues = NumRowDoubleArray()
        rangeValues = NumRowDoubleArray()
        rowNames = NumRowStrArray()
        rowType = NumRowCharArray()
        self.c2n = {}
        self.n2c = {}
        i = 0
        for c in lp.constraints:
            rhsValues[i] = -lp.constraints[c].constant
            rangeValues[i] = 0.0
            rowNames[i] = to_string(c)
            rowType[i] = to_string(senseDict[lp.constraints[c].sense])
            self.c2n[c] = i
            self.n2c[i] = c
            i = i + 1
        coeffs = lp.coefficients()
        sparseMatrix = sparse.Matrix(list(range(numRows)), list(range(numVars)))
        for var, row, coeff in coeffs:
            sparseMatrix.add(self.c2n[row], self.vname2n[var], coeff)
        numels, mystartsBase, mylenBase, myindBase, myelemBase = sparseMatrix.col_based_arrays()
        elemBase = ctypesArrayFill(myelemBase, ctypes.c_double)
        indBase = ctypesArrayFill(myindBase, ctypes.c_int)
        startsBase = ctypesArrayFill(mystartsBase, ctypes.c_int)
        lenBase = ctypesArrayFill(mylenBase, ctypes.c_int)
        NumVarCharArray = ctypes.c_char * numVars
        columnType = NumVarCharArray()
        if lp.isMIP():
            for v in lp.variables():
                columnType[self.v2n[v]] = to_string(LpVarCategories[v.cat])
        self.addedVars = numVars
        self.addedRows = numRows
        return (numVars, numRows, numels, rangeCount, objSense, objectCoeffs, objectConst, rhsValues, rangeValues, rowType, startsBase, lenBase, indBase, elemBase, lowerBounds, upperBounds, initValues, colNames, rowNames, columnType, self.n2v, self.n2c)

    def toDict(self):
        data = dict(solver=self.name)
        for k in ['mip', 'msg', 'keepFiles']:
            try:
                data[k] = getattr(self, k)
            except AttributeError:
                pass
        for k in ['timeLimit', 'options']:
            try:
                value = getattr(self, k)
                if value:
                    data[k] = value
            except AttributeError:
                pass
        data.update(self.optionsDict)
        return data
    to_dict = toDict

    def toJson(self, filename, *args, **kwargs):
        with open(filename, 'w') as f:
            json.dump(self.toDict(), f, *args, **kwargs)
    to_json = toJson