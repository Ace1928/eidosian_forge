import numpy as np
import os
import time
from ..utils import *
def _4SS(imgP, imgI, mbSize, p):
    h, w = imgP.shape
    vectors = np.zeros((np.int(h / mbSize), np.int(w / mbSize), 2))
    costs = np.ones((3, 3), dtype=np.float32) * 65537
    computations = 0
    for i in range(0, h - mbSize + 1, mbSize):
        for j in range(0, w - mbSize + 1, mbSize):
            x = j
            y = i
            costs[1, 1] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[i:i + mbSize, j:j + mbSize])
            computations += 1
            for m in range(-2, 3, 2):
                for n in range(-2, 3, 2):
                    refBlkVer = y + m
                    refBlkHor = x + n
                    if not _checkBounded(refBlkHor, refBlkVer, w, h, mbSize):
                        continue
                    costRow = np.int(m / 2 + 1)
                    costCol = np.int(n / 2 + 1)
                    if costRow == 1 and costCol == 1:
                        continue
                    costs[costRow, costCol] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize])
                    computations = computations + 1
            dx, dy, mi = _minCost(costs)
            flag_4ss = 0
            if dx == 1 and dy == 1:
                flag_4ss = 1
            else:
                xLast = x
                yLast = y
                x += (dx - 1) * 2
                y += (dy - 1) * 2
            costs[:, :] = 65537
            costs[1, 1] = mi
            stage = 1
            while flag_4ss == 0 and stage <= 2:
                for m in range(-2, 3, 2):
                    for n in range(-2, 3, 2):
                        refBlkVer = y + m
                        refBlkHor = x + n
                        if not _checkBounded(refBlkHor, refBlkVer, w, h, mbSize):
                            continue
                        if refBlkHor >= xLast - 2 and refBlkHor <= xLast + 2 and (refBlkVer >= yLast - 2) and (refBlkVer >= yLast + 2):
                            continue
                        costRow = np.int(m / 2) + 1
                        costCol = np.int(n / 2) + 1
                        if costRow == 1 and costCol == 1:
                            continue
                        costs[costRow, costCol] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize])
                        computations += 1
                dx, dy, mi = _minCost(costs)
                if dx == 1 and dy == 1:
                    flag_4ss = 1
                else:
                    flag_4ss = 0
                    xLast = x
                    yLast = y
                    x = x + (dx - 1) * 2
                    y = y + (dy - 1) * 2
                costs[:, :] = 65537
                costs[1, 1] = mi
                stage += 1
            for m in range(-1, 2):
                for n in range(-1, 2):
                    refBlkVer = y + m
                    refBlkHor = x + n
                    if not _checkBounded(refBlkHor, refBlkVer, w, h, mbSize):
                        continue
                    costRow = m + 1
                    costRow = n + 1
                    if costRow == 2 and costCol == 2:
                        continue
                    costs[costRow, costCol] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize])
                    computations += 1
            dx, dy, mi = _minCost(costs)
            x += dx - 1
            y += dy - 1
            vectors[np.int(i / mbSize), np.int(j / mbSize), :] = [y - i, x - j]
            costs[:, :] = 65537
    return (vectors, computations / (h * w / mbSize ** 2))