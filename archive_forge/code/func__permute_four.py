import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def _permute_four(self, cases):
    case1, case2, case3, case4 = cases
    permutations = []
    permutations.append([case1, case2, case3, case4])
    permutations.append([case1, case2, case4, case3])
    permutations.append([case1, case3, case2, case4])
    permutations.append([case1, case3, case4, case2])
    permutations.append([case1, case4, case2, case3])
    permutations.append([case1, case4, case3, case2])
    permutations.append([case2, case1, case3, case4])
    permutations.append([case2, case1, case4, case3])
    permutations.append([case2, case3, case1, case4])
    permutations.append([case2, case3, case4, case1])
    permutations.append([case2, case4, case1, case3])
    permutations.append([case2, case4, case3, case1])
    permutations.append([case3, case2, case1, case4])
    permutations.append([case3, case2, case4, case1])
    permutations.append([case3, case1, case2, case4])
    permutations.append([case3, case1, case4, case2])
    permutations.append([case3, case4, case2, case1])
    permutations.append([case3, case4, case1, case2])
    permutations.append([case4, case2, case3, case1])
    permutations.append([case4, case2, case1, case3])
    permutations.append([case4, case3, case2, case1])
    permutations.append([case4, case3, case1, case2])
    permutations.append([case4, case1, case2, case3])
    permutations.append([case4, case1, case3, case2])
    return permutations