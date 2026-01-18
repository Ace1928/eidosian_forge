from collections import deque
from functools import reduce
from math import ceil, floor
import operator
import re
from itertools import chain
import six
from genshi.compat import IS_PYTHON2
from genshi.core import Stream, Attrs, Namespace, QName
from genshi.core import START, END, TEXT, START_NS, END_NS, COMMENT, PI, \
class SimplePathStrategy(object):
    """Strategy for path with only local names, attributes and text nodes."""

    @classmethod
    def supports(cls, path):
        if path[0][0] is ATTRIBUTE:
            return False
        allowed_tests = (LocalNameTest, CommentNodeTest, TextNodeTest)
        for _, nodetest, predicates in path:
            if predicates:
                return False
            if not isinstance(nodetest, allowed_tests):
                return False
        return True

    def __init__(self, path):
        self.fragments = []
        self_beginning = False
        fragment = []

        def nodes_equal(node1, node2):
            """Tests if two node tests are equal"""
            if type(node1) is not type(node2):
                return False
            if type(node1) == LocalNameTest:
                return node1.name == node2.name
            return True

        def calculate_pi(f):
            """KMP prefix calculation for table"""
            if len(f) == 0:
                return []
            pi = [0]
            s = 0
            for i in range(1, len(f)):
                while s > 0 and (not nodes_equal(f[s], f[i])):
                    s = pi[s - 1]
                if nodes_equal(f[s], f[i]):
                    s += 1
                pi.append(s)
            return pi
        for axis in path:
            if axis[0] is SELF:
                if len(fragment) != 0:
                    if axis[1] != fragment[-1][1]:
                        self.fragments = None
                        return
                else:
                    self_beginning = True
                    fragment.append(axis[1])
            elif axis[0] is CHILD:
                fragment.append(axis[1])
            elif axis[0] is ATTRIBUTE:
                pi = calculate_pi(fragment)
                self.fragments.append((fragment, pi, axis[1], self_beginning))
                return
            else:
                pi = calculate_pi(fragment)
                self.fragments.append((fragment, pi, None, self_beginning))
                fragment = [axis[1]]
                if axis[0] is DESCENDANT:
                    self_beginning = False
                else:
                    self_beginning = True
        pi = calculate_pi(fragment)
        self.fragments.append((fragment, pi, None, self_beginning))

    def test(self, ignore_context):
        stack = []
        stack_push = stack.append
        stack_pop = stack.pop
        frags = self.fragments
        frags_len = len(frags)

        def _test(event, namespaces, variables, updateonly=False):
            if frags is None:
                return None
            kind, data, pos = event[:3]
            if kind is END:
                if stack:
                    stack_pop()
                return None
            if kind is START_NS or kind is END_NS or kind is START_CDATA or (kind is END_CDATA):
                return None
            if not stack:
                fid = 0
                while not frags[fid][0]:
                    fid += 1
                p = 0
                ic = ignore_context or fid > 0
                if not frags[fid][3] and (not ignore_context or fid > 0):
                    stack_push((fid, p, ic))
                    return None
            else:
                fid, p, ic = stack[-1]
            if fid is not None and (not ic):
                frag, pi, attrib, _ = frags[fid]
                frag_len = len(frag)
                if p == frag_len:
                    pass
                elif frag[p](kind, data, pos, namespaces, variables):
                    p += 1
                else:
                    fid, p = (None, None)
                if p == frag_len and fid + 1 != frags_len:
                    fid += 1
                    p = 0
                    ic = True
            if fid is None:
                if kind is START:
                    stack_push((fid, p, ic))
                return None
            if ic:
                while True:
                    frag, pi, attrib, _ = frags[fid]
                    frag_len = len(frag)
                    while p > 0 and (p >= frag_len or not frag[p](kind, data, pos, namespaces, variables)):
                        p = pi[p - 1]
                    if frag[p](kind, data, pos, namespaces, variables):
                        p += 1
                    if p == frag_len:
                        if fid + 1 == frags_len:
                            break
                        else:
                            fid += 1
                            p = 0
                            ic = True
                            if not frags[fid][3]:
                                break
                    else:
                        break
            if kind is START:
                if not ic and fid + 1 == frags_len and (p == frag_len):
                    stack_push((None, None, ic))
                else:
                    stack_push((fid, p, ic))
            if fid + 1 == frags_len and p == frag_len:
                if attrib:
                    return attrib(kind, data, pos, namespaces, variables)
                return True
            return None
        return _test