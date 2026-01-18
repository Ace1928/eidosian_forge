import functools
import logging
import math
import operator
import sympy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import torch
import torch.fx
import torch.fx.traceback as fx_traceback
from torch._dynamo.exc import TorchDynamoException
from torch.fx.node import Argument, Target
from torch.utils._sympy.interp import sympy_interp
from torch.fx.experimental import _config as config
class TranslationValidator:

    def __init__(self) -> None:
        log.debug('new instance')
        self.symbols: Dict[sympy.Symbol, z3.ExprRef] = {}
        self._source_exprs: Set[z3.BoolRef] = set()
        self._target_exprs: Set[z3.BoolRef] = set()
        self._assertions: Set[z3.BoolRef] = set()

    def z3var(self, symbol: sympy.Symbol) -> z3.ExprRef:
        assert symbol in self.symbols, f'Z3 variable not found for: {symbol}'
        return self.symbols[symbol]

    def add_var(self, symbol: sympy.Symbol, type: Type) -> z3.ExprRef:
        if symbol in self.symbols:
            return self.symbols[symbol]
        log.debug('new variable: %s (%s)', symbol.name, type.__name__)
        if type is int:
            var = z3.Int(symbol.name)
            if symbol.is_positive:
                self._target_exprs.add(var > 0)
        elif type is float:
            var = z3.Real(symbol.name)
        elif type is bool:
            var = z3.Bool(symbol.name)
        else:
            raise RuntimeError(f'unsupported type for Z3 variable: {type}')
        self.symbols[symbol] = var
        return var

    def _check_freesymbols(self, e: sympy.Basic) -> None:
        for s in e.free_symbols:
            assert isinstance(s, sympy.Symbol)
            self.z3var(s)

    def to_z3_boolean_expr(self, e: sympy.Basic) -> z3.BoolRef:
        z3expr = SympyToZ3(self).run(e)
        assert isinstance(z3expr, z3.BoolRef), f'expected boolean expression. Got: {z3expr}'
        return z3expr

    def add_source_expr(self, e: z3.BoolRef) -> None:
        if e not in self._source_exprs:
            log.debug('add source guard: %s', z3str(e))
        self._source_exprs.add(e)

    def add_target_expr(self, e: sympy.Expr) -> None:
        self._check_freesymbols(e)
        z3expr = self.to_z3_boolean_expr(e)
        if e not in self._target_exprs:
            log.debug('add target guard: %s', z3str(z3expr))
        self._target_exprs.add(z3expr)

    def add_assertion(self, e: Union[z3.BoolRef, sympy.Basic]) -> None:
        if isinstance(e, sympy.Basic):
            self._check_freesymbols(e)
            ref = self.to_z3_boolean_expr(e)
        else:
            ref = e
        assert isinstance(ref, z3.BoolRef)
        if ref not in self._assertions:
            log.debug('add assertion: %s', z3str(ref))
        self._assertions.add(ref)

    def validate(self) -> None:
        from torch._dynamo.utils import dynamo_timed
        if len(self._source_exprs) == 0 or len(self._target_exprs) == 0:
            return None
        solver = z3.SolverFor('QF_NRA')
        solver.set(timeout=translation_validation_timeout())
        for assertion in self._assertions:
            solver.add(assertion)
        solver.add(z3.Not(z3.And(*self._source_exprs)))
        solver.add(*self._target_exprs)
        log.debug('translation validation: start')
        r = dynamo_timed()(solver.check)()
        if r == z3.sat:
            model = solver.model()
            raise ValidationException(model, self._assertions, self._target_exprs, failed_source_exprs=[inp for inp in self._source_exprs if not model.evaluate(inp)])
        elif r == z3.unknown:
            log.warning('translation validation: could not validate: got z3.unknown')
        else:
            assert r == z3.unsat
            log.debug('translation validation: success')