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
class GenericStrategy(object):

    @classmethod
    def supports(cls, path):
        return True

    def __init__(self, path):
        self.path = path

    def test(self, ignore_context):
        p = self.path
        if ignore_context:
            if p[0][0] is ATTRIBUTE:
                steps = [_DOTSLASHSLASH] + p
            else:
                steps = [(DESCENDANT_OR_SELF, p[0][1], p[0][2])] + p[1:]
        elif p[0][0] is CHILD or p[0][0] is ATTRIBUTE or p[0][0] is DESCENDANT:
            steps = [_DOTSLASH] + p
        else:
            steps = p
        stack = [[(0, [[]])]]

        def _test(event, namespaces, variables, updateonly=False):
            kind, data, pos = event[:3]
            retval = None
            if kind is END:
                if stack:
                    stack.pop()
                return None
            if kind is START_NS or kind is END_NS or kind is START_CDATA or (kind is END_CDATA):
                return None
            pos_queue = deque([(pos, cou, []) for pos, cou in stack[-1]])
            next_pos = []
            real_len = len(steps) - (steps[-1][0] == ATTRIBUTE or (1 and 0))
            last_checked = -1
            while pos_queue:
                x, pcou, mcou = pos_queue.popleft()
                axis, nodetest, predicates = steps[x]
                if (axis is DESCENDANT or axis is DESCENDANT_OR_SELF) and pcou:
                    if next_pos and next_pos[-1][0] == x:
                        next_pos[-1][1].extend(pcou)
                    else:
                        next_pos.append((x, pcou))
                if not nodetest(kind, data, pos, namespaces, variables):
                    continue
                missed = set()
                counters_len = len(pcou) + len(mcou)
                cnum = 0
                matched = True
                if predicates:
                    for predicate in predicates:
                        pretval = predicate(kind, data, pos, namespaces, variables)
                        if type(pretval) is float:
                            for i, cou in enumerate(chain(pcou, mcou)):
                                if i in missed:
                                    continue
                                if len(cou) < cnum + 1:
                                    cou.append(0)
                                cou[cnum] += 1
                                if cou[cnum] != int(pretval):
                                    missed.add(i)
                            if len(missed) == counters_len:
                                pretval = False
                            cnum += 1
                        if not pretval:
                            matched = False
                            break
                if not matched:
                    continue
                child_counter = []
                if x + 1 == real_len:
                    matched = True
                    axis, nodetest, predicates = steps[-1]
                    if axis is ATTRIBUTE:
                        matched = nodetest(kind, data, pos, namespaces, variables)
                    if matched:
                        retval = matched
                else:
                    next_axis = steps[x + 1][0]
                    if next_axis is DESCENDANT_OR_SELF or next_axis is SELF:
                        if not pos_queue or pos_queue[0][0] > x + 1:
                            pos_queue.appendleft((x + 1, [], [child_counter]))
                        else:
                            pos_queue[0][2].append(child_counter)
                    if next_axis is not SELF:
                        next_pos.append((x + 1, [child_counter]))
            if kind is START:
                stack.append(next_pos)
            return retval
        return _test