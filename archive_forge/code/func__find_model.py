from collections import defaultdict
from heapq import heappush, heappop
from sympy.core.sorting import ordered
from sympy.assumptions.cnf import EncodedCNF
def _find_model(self):
    """
        Main DPLL loop. Returns a generator of models.

        Variables are chosen successively, and assigned to be either
        True or False. If a solution is not found with this setting,
        the opposite is chosen and the search continues. The solver
        halts when every variable has a setting.

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set())
        >>> list(l._find_model())
        [{1: True, 2: False, 3: False}, {1: True, 2: True, 3: True}]

        >>> from sympy.abc import A, B, C
        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set(), [A, B, C])
        >>> list(l._find_model())
        [{A: True, B: False, C: False}, {A: True, B: True, C: True}]

        """
    flip_var = False
    self._simplify()
    if self.is_unsatisfied:
        return
    while True:
        if self.num_decisions % self.INTERVAL == 0:
            for func in self.update_functions:
                func()
        if flip_var:
            flip_var = False
            lit = self._current_level.decision
        else:
            lit = self.heur_calculate()
            self.num_decisions += 1
            if 0 == lit:
                yield {self.symbols[abs(lit) - 1]: lit > 0 for lit in self.var_settings}
                while self._current_level.flipped:
                    self._undo()
                if len(self.levels) == 1:
                    return
                flip_lit = -self._current_level.decision
                self._undo()
                self.levels.append(Level(flip_lit, flipped=True))
                flip_var = True
                continue
            self.levels.append(Level(lit))
        self._assign_literal(lit)
        self._simplify()
        if self.is_unsatisfied:
            self.is_unsatisfied = False
            while self._current_level.flipped:
                self._undo()
                if 1 == len(self.levels):
                    return
            self.add_learned_clause(self.compute_conflict())
            flip_lit = -self._current_level.decision
            self._undo()
            self.levels.append(Level(flip_lit, flipped=True))
            flip_var = True