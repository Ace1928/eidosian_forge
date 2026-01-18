from functools import reduce
import copy
import math
import random
import sys
import warnings
from Bio import File
from Bio.Data import IUPACData
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning, BiopythonWarning
from Bio.Nexus.StandardData import StandardData
from Bio.Nexus.Trees import Tree
def _compact4nexus(orig_list):
    """Compact lists for Nexus output (PRIVATE).

    Example
    -------
    >>> _compact4nexus([1, 2, 3, 5, 6, 7, 8, 12, 15, 18, 20])
    '2-4 6-9 13-19\\\\3 21'

    Transform [1 2 3 5 6 7 8 12 15 18 20] (baseindex 0, used in the Nexus class)
    into '2-4 6-9 13-19\\\\3 21' (baseindex 1, used in programs like Paup or MrBayes.).

    """
    if not orig_list:
        return ''
    orig_list = sorted(set(orig_list))
    shortlist = []
    clist = orig_list[:]
    clist.append(clist[-1] + 0.5)
    while len(clist) > 1:
        step = 1
        for i, x in enumerate(clist):
            if x == clist[0] + i * step:
                continue
            elif i == 1 and len(clist) > 3 and (clist[i + 1] - x == x - clist[0]):
                step = x - clist[0]
            else:
                sub = clist[:i]
                if len(sub) == 1:
                    shortlist.append(str(sub[0] + 1))
                elif step == 1:
                    shortlist.append('%d-%d' % (sub[0] + 1, sub[-1] + 1))
                else:
                    shortlist.append('%d-%d\\%d' % (sub[0] + 1, sub[-1] + 1, step))
                clist = clist[i:]
                break
    return ' '.join(shortlist)