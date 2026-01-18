import numpy as np
import os
import time
from ..utils import *
def _ARPS(imgP, imgI, mbSize, p):
    h, w = imgP.shape
    vectors = np.zeros((np.int(h / mbSize), np.int(w / mbSize), 2))
    costs = np.ones(6) * 65537
    SDSP = []
    SDSP.append([0, -1])
    SDSP.append([-1, 0])
    SDSP.append([0, 0])
    SDSP.append([1, 0])
    SDSP.append([0, 1])
    LDSP = {}
    checkMatrix = np.zeros((2 * p + 1, 2 * p + 1))
    computations = 0
    for i in range(0, h - mbSize + 1, mbSize):
        for j in range(0, w - mbSize + 1, mbSize):
            x = j
            y = i
            costs[2] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[i:i + mbSize, j:j + mbSize])
            checkMatrix[p, p] = 1
            computations += 1
            if j == 0:
                stepSize = 2
                maxIndex = 5
            else:
                u = vectors[np.int(i / mbSize), np.int(j / mbSize) - 1, 0]
                v = vectors[np.int(i / mbSize), np.int(j / mbSize) - 1, 1]
                stepSize = np.int(np.max((np.abs(u), np.abs(v))))
                if np.abs(u) == stepSize and np.abs(v) == 0 or (np.abs(v) == stepSize and np.abs(u) == 0):
                    maxIndex = 5
                else:
                    maxIndex = 6
                    LDSP[5] = [np.int(v), np.int(u)]
            LDSP[0] = [0, -stepSize]
            LDSP[1] = [-stepSize, 0]
            LDSP[2] = [0, 0]
            LDSP[3] = [stepSize, 0]
            LDSP[4] = [0, stepSize]
            for k in range(maxIndex):
                refBlkVer = y + LDSP[k][1]
                refBlkHor = x + LDSP[k][0]
                if not _checkBounded(refBlkHor, refBlkVer, w, h, mbSize):
                    continue
                if k == 2 or stepSize == 0:
                    continue
                costs[k] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize])
                computations += 1
                checkMatrix[LDSP[k][1] + p, LDSP[k][0] + p] = 1
            if costs[2] != 0:
                point = np.argmin(costs)
                cost = costs[point]
            else:
                point = 2
                cost = costs[point]
            x += LDSP[point][0]
            y += LDSP[point][1]
            costs[:] = 65537
            costs[2] = cost
            doneFlag = 0
            while doneFlag == 0:
                for k in range(5):
                    refBlkVer = y + SDSP[k][1]
                    refBlkHor = x + SDSP[k][0]
                    if not _checkBounded(refBlkHor, refBlkVer, w, h, mbSize):
                        continue
                    if k == 2:
                        continue
                    elif refBlkHor < j - p or refBlkHor > j + p or refBlkVer < i - p or (refBlkVer > i + p):
                        continue
                    elif checkMatrix[y - i + SDSP[k][1] + p, x - j + SDSP[k][0] + p] == 1:
                        continue
                    costs[k] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize])
                    checkMatrix[y - i + SDSP[k][1] + p, x - j + SDSP[k][0] + p] = 1
                    computations += 1
                if costs[2] != 0:
                    point = np.argmin(costs)
                    cost = costs[point]
                else:
                    point = 2
                    cost = costs[point]
                doneFlag = 1
                if point != 2:
                    doneFlag = 0
                    y += SDSP[point][1]
                    x += SDSP[point][0]
                    costs[:] = 65537
                    costs[2] = cost
            vectors[np.int(i / mbSize), np.int(j / mbSize), :] = [x - j, y - i]
            costs[:] = 65537
            checkMatrix[:, :] = 0
    return (vectors, computations / (h * w / mbSize ** 2))