import copy
import math
import copyreg
import random
import re
import sys
import types
import warnings
from collections import defaultdict, deque
from functools import partial, wraps
from operator import eq, lt
from . import tools  # Needed by HARM-GP
def _genpop(n, pickfrom=[], acceptfunc=lambda s: True, producesizes=False):
    producedpop = []
    producedpopsizes = []
    while len(producedpop) < n:
        if len(pickfrom) > 0:
            aspirant = pickfrom.pop()
            if acceptfunc(len(aspirant)):
                producedpop.append(aspirant)
                if producesizes:
                    producedpopsizes.append(len(aspirant))
        else:
            opRandom = random.random()
            if opRandom < cxpb:
                aspirant1, aspirant2 = toolbox.mate(*map(toolbox.clone, toolbox.select(population, 2)))
                del aspirant1.fitness.values, aspirant2.fitness.values
                if acceptfunc(len(aspirant1)):
                    producedpop.append(aspirant1)
                    if producesizes:
                        producedpopsizes.append(len(aspirant1))
                if len(producedpop) < n and acceptfunc(len(aspirant2)):
                    producedpop.append(aspirant2)
                    if producesizes:
                        producedpopsizes.append(len(aspirant2))
            else:
                aspirant = toolbox.clone(toolbox.select(population, 1)[0])
                if opRandom - cxpb < mutpb:
                    aspirant = toolbox.mutate(aspirant)[0]
                    del aspirant.fitness.values
                if acceptfunc(len(aspirant)):
                    producedpop.append(aspirant)
                    if producesizes:
                        producedpopsizes.append(len(aspirant))
    if producesizes:
        return (producedpop, producedpopsizes)
    else:
        return producedpop