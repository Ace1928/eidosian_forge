from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError, clock, log
from .core import cbc_path, pulp_cbc_path, coinMP_path, devnull, operating_system
import os
from .. import constants
from tempfile import mktemp
import ctypes
import warnings
class COINMP_DLL(LpSolver):
    """
    The COIN_MP LP MIP solver (via a DLL or linux so)

    :param timeLimit: The number of seconds before forcing the solver to exit
    :param epgap: The fractional mip tolerance
    """
    name = 'COINMP_DLL'
    try:
        lib = COINMP_DLL_load_dll(coinMP_path)
    except (ImportError, OSError):

        @classmethod
        def available(cls):
            """True if the solver is available"""
            return False

        def actualSolve(self, lp):
            """Solve a well formulated lp problem"""
            raise PulpSolverError('COINMP_DLL: Not Available')
    else:
        COIN_INT_LOGLEVEL = 7
        COIN_REAL_MAXSECONDS = 16
        COIN_REAL_MIPMAXSEC = 19
        COIN_REAL_MIPFRACGAP = 34
        lib.CoinGetInfinity.restype = ctypes.c_double
        lib.CoinGetVersionStr.restype = ctypes.c_char_p
        lib.CoinGetSolutionText.restype = ctypes.c_char_p
        lib.CoinGetObjectValue.restype = ctypes.c_double
        lib.CoinGetMipBestBound.restype = ctypes.c_double

        def __init__(self, cuts=1, presolve=1, dual=1, crash=0, scale=1, rounding=1, integerPresolve=1, strong=5, *args, **kwargs):
            LpSolver.__init__(self, *args, **kwargs)
            self.fracGap = None
            gapRel = self.optionsDict.get('gapRel')
            if gapRel is not None:
                self.fracGap = float(gapRel)
            if self.timeLimit is not None:
                self.timeLimit = float(self.timeLimit)
            self.cuts = cuts
            self.presolve = presolve
            self.dual = dual
            self.crash = crash
            self.scale = scale
            self.rounding = rounding
            self.integerPresolve = integerPresolve
            self.strong = strong

        def copy(self):
            """Make a copy of self"""
            aCopy = LpSolver.copy(self)
            aCopy.cuts = self.cuts
            aCopy.presolve = self.presolve
            aCopy.dual = self.dual
            aCopy.crash = self.crash
            aCopy.scale = self.scale
            aCopy.rounding = self.rounding
            aCopy.integerPresolve = self.integerPresolve
            aCopy.strong = self.strong
            return aCopy

        @classmethod
        def available(cls):
            """True if the solver is available"""
            return True

        def getSolverVersion(self):
            """
            returns a solver version string

            example:
            >>> COINMP_DLL().getSolverVersion() # doctest: +ELLIPSIS
            '...'
            """
            return self.lib.CoinGetVersionStr()

        def actualSolve(self, lp):
            """Solve a well formulated lp problem"""
            self.debug = 0
            self.lib.CoinInitSolver('')
            self.hProb = hProb = self.lib.CoinCreateProblem(lp.name)
            self.lib.CoinSetIntOption(hProb, self.COIN_INT_LOGLEVEL, ctypes.c_int(self.msg))
            if self.timeLimit:
                if self.mip:
                    self.lib.CoinSetRealOption(hProb, self.COIN_REAL_MIPMAXSEC, ctypes.c_double(self.timeLimit))
                else:
                    self.lib.CoinSetRealOption(hProb, self.COIN_REAL_MAXSECONDS, ctypes.c_double(self.timeLimit))
            if self.fracGap:
                self.lib.CoinSetRealOption(hProb, self.COIN_REAL_MIPFRACGAP, ctypes.c_double(self.fracGap))
            coinDblMax = self.lib.CoinGetInfinity()
            if self.debug:
                print('Before getCoinMPArrays')
            numVars, numRows, numels, rangeCount, objectSense, objectCoeffs, objectConst, rhsValues, rangeValues, rowType, startsBase, lenBase, indBase, elemBase, lowerBounds, upperBounds, initValues, colNames, rowNames, columnType, n2v, n2c = self.getCplexStyleArrays(lp)
            self.lib.CoinLoadProblem(hProb, numVars, numRows, numels, rangeCount, objectSense, objectConst, objectCoeffs, lowerBounds, upperBounds, rowType, rhsValues, rangeValues, startsBase, lenBase, indBase, elemBase, colNames, rowNames, 'Objective')
            if lp.isMIP() and self.mip:
                self.lib.CoinLoadInteger(hProb, columnType)
            if self.msg == 0:
                self.lib.CoinRegisterMsgLogCallback(hProb, ctypes.c_char_p(''), ctypes.POINTER(ctypes.c_int)())
            self.coinTime = -clock()
            self.lib.CoinOptimizeProblem(hProb, 0)
            self.coinTime += clock()
            CoinLpStatus = {0: constants.LpStatusOptimal, 1: constants.LpStatusInfeasible, 2: constants.LpStatusInfeasible, 3: constants.LpStatusNotSolved, 4: constants.LpStatusNotSolved, 5: constants.LpStatusNotSolved, -1: constants.LpStatusUndefined}
            solutionStatus = self.lib.CoinGetSolutionStatus(hProb)
            solutionText = self.lib.CoinGetSolutionText(hProb)
            objectValue = self.lib.CoinGetObjectValue(hProb)
            NumVarDoubleArray = ctypes.c_double * numVars
            NumRowsDoubleArray = ctypes.c_double * numRows
            cActivity = NumVarDoubleArray()
            cReducedCost = NumVarDoubleArray()
            cSlackValues = NumRowsDoubleArray()
            cShadowPrices = NumRowsDoubleArray()
            self.lib.CoinGetSolutionValues(hProb, ctypes.byref(cActivity), ctypes.byref(cReducedCost), ctypes.byref(cSlackValues), ctypes.byref(cShadowPrices))
            variablevalues = {}
            variabledjvalues = {}
            constraintpivalues = {}
            constraintslackvalues = {}
            if lp.isMIP() and self.mip:
                lp.bestBound = self.lib.CoinGetMipBestBound(hProb)
            for i in range(numVars):
                variablevalues[self.n2v[i].name] = cActivity[i]
                variabledjvalues[self.n2v[i].name] = cReducedCost[i]
            lp.assignVarsVals(variablevalues)
            lp.assignVarsDj(variabledjvalues)
            for i in range(numRows):
                constraintpivalues[self.n2c[i]] = cShadowPrices[i]
                constraintslackvalues[self.n2c[i]] = cSlackValues[i]
            lp.assignConsPi(constraintpivalues)
            lp.assignConsSlack(constraintslackvalues)
            self.lib.CoinFreeSolver()
            status = CoinLpStatus[self.lib.CoinGetSolutionStatus(hProb)]
            lp.assignStatus(status)
            return status