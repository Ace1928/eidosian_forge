from collections import defaultdict
import networkx as nx
def add_constraints(self, ei, e):
    P = ConflictPair()
    while True:
        Q = self.S.pop()
        if not Q.left.empty():
            Q.swap()
        if not Q.left.empty():
            return False
        if self.lowpt[Q.right.low] > self.lowpt[e]:
            if P.right.empty():
                P.right = Q.right.copy()
            else:
                self.ref[P.right.low] = Q.right.high
            P.right.low = Q.right.low
        else:
            self.ref[Q.right.low] = self.lowpt_edge[e]
        if top_of_stack(self.S) == self.stack_bottom[ei]:
            break
    while top_of_stack(self.S).left.conflicting(ei, self) or top_of_stack(self.S).right.conflicting(ei, self):
        Q = self.S.pop()
        if Q.right.conflicting(ei, self):
            Q.swap()
        if Q.right.conflicting(ei, self):
            return False
        self.ref[P.right.low] = Q.right.high
        if Q.right.low is not None:
            P.right.low = Q.right.low
        if P.left.empty():
            P.left = Q.left.copy()
        else:
            self.ref[P.left.low] = Q.left.high
        P.left.low = Q.left.low
    if not (P.left.empty() and P.right.empty()):
        self.S.append(P)
    return True