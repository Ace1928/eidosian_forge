import unittest
from numba.tests.support import TestCase
from numba.core.compiler import run_frontend
class TestByteFlowIssues(TestCase):

    def test_issue_5087(self):

        def udt():
            print
            print
            print
            for i in range:
                print
                print
                print
                print
                print
                print
                print
                print
                print
                print
                print
                print
                print
                print
                print
                print
                print
                print
                for j in range:
                    print
                    print
                    print
                    print
                    print
                    print
                    print
                    for k in range:
                        for l in range:
                            print
                    print
                    print
                    print
                    print
                    print
                    print
                    print
                    print
                    print
                    if print:
                        for n in range:
                            print
                    else:
                        print
        run_frontend(udt)

    def test_issue_5097(self):

        def udt():
            for i in range(0):
                if i > 0:
                    pass
                a = None
        run_frontend(udt)

    def test_issue_5680(self):

        def udt():
            for k in range(0):
                if 1 == 1:
                    ...
                if 'a' == 'a':
                    ...
        run_frontend(udt)