import numpy as np
import os
import time
from ..utils import *
def _DS(imgP, imgI, mbSize, p):
    h, w = imgP.shape
    vectors = np.zeros((np.int(h / mbSize), np.int(w / mbSize), 2))
    costs = np.ones(9) * 65537
    L = np.floor(np.log2(p + 1))
    LDSP = []
    LDSP.append([0, -2])
    LDSP.append([-1, -1])
    LDSP.append([1, -1])
    LDSP.append([-2, 0])
    LDSP.append([0, 0])
    LDSP.append([2, 0])
    LDSP.append([-1, 1])
    LDSP.append([1, 1])
    LDSP.append([0, 2])
    SDSP = []
    SDSP.append([0, -1])
    SDSP.append([-1, 0])
    SDSP.append([0, 0])
    SDSP.append([1, 0])
    SDSP.append([0, 1])
    computations = 0
    for i in range(0, h - mbSize + 1, mbSize):
        for j in range(0, w - mbSize + 1, mbSize):
            x = j
            y = i
            costs[4] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[i:i + mbSize, j:j + mbSize])
            cost = 0
            point = 4
            if costs[4] != 0:
                computations += 1
                for k in range(9):
                    refBlkVer = y + LDSP[k][1]
                    refBlkHor = x + LDSP[k][0]
                    if not _checkBounded(refBlkHor, refBlkVer, w, h, mbSize):
                        continue
                    if k == 4:
                        continue
                    costs[k] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize])
                    computations += 1
                point = np.argmin(costs)
                cost = costs[point]
            SDSPFlag = 1
            if point != 4:
                SDSPFlag = 0
                cornerFlag = 1
                if np.abs(LDSP[point][0]) == np.abs(LDSP[point][1]):
                    cornerFlag = 0
                xLast = x
                yLast = y
                x = x + LDSP[point][0]
                y = y + LDSP[point][1]
                costs[:] = 65537
                costs[4] = cost
            while SDSPFlag == 0:
                if cornerFlag == 1:
                    for k in range(9):
                        refBlkVer = y + LDSP[k][1]
                        refBlkHor = x + LDSP[k][0]
                        if not _checkBounded(refBlkHor, refBlkVer, w, h, mbSize):
                            continue
                        if k == 4:
                            continue
                        if refBlkHor >= xLast - 1 and refBlkHor <= xLast + 1 and (refBlkVer >= yLast - 1) and (refBlkVer <= yLast + 1):
                            continue
                        elif refBlkHor < j - p or refBlkHor > j + p or refBlkVer < i - p or (refBlkVer > i + p):
                            continue
                        else:
                            costs[k] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize])
                            computations += 1
                else:
                    lst = []
                    if point == 1:
                        lst = np.array([0, 1, 3])
                    elif point == 2:
                        lst = np.array([0, 2, 5])
                    elif point == 6:
                        lst = np.array([3, 6, 8])
                    elif point == 7:
                        lst = np.array([5, 7, 8])
                    for idx in lst:
                        refBlkVer = y + LDSP[idx][1]
                        refBlkHor = x + LDSP[idx][0]
                        if not _checkBounded(refBlkHor, refBlkVer, w, h, mbSize):
                            continue
                        elif refBlkHor < j - p or refBlkHor > j + p or refBlkVer < i - p or (refBlkVer > i + p):
                            continue
                        else:
                            costs[idx] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize])
                            computations += 1
                point = np.argmin(costs)
                cost = costs[point]
                SDSPFlag = 1
                if point != 4:
                    SDSPFlag = 0
                    cornerFlag = 1
                    if np.abs(LDSP[point][0]) == np.abs(LDSP[point][1]):
                        cornerFlag = 0
                    xLast = x
                    yLast = y
                    x += LDSP[point][0]
                    y += LDSP[point][1]
                    costs[:] = 65537
                    costs[4] = cost
            costs[:] = 65537
            costs[2] = cost
            for k in range(5):
                refBlkVer = y + SDSP[k][1]
                refBlkHor = x + SDSP[k][0]
                if not _checkBounded(refBlkHor, refBlkVer, w, h, mbSize):
                    continue
                elif refBlkHor < j - p or refBlkHor > j + p or refBlkVer < i - p or (refBlkVer > i + p):
                    continue
                if k == 2:
                    continue
                costs[k] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize])
                computations += 1
            point = 2
            cost = 0
            if costs[2] != 0:
                point = np.argmin(costs)
                cost = costs[point]
            x += SDSP[point][0]
            y += SDSP[point][1]
            vectors[np.int(i / mbSize), np.int(j / mbSize), :] = [x - j, y - i]
            costs[:] = 65537
    return (vectors, computations / (h * w / mbSize ** 2))