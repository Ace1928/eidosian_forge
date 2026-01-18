from __future__ import annotations
import ast
import builtins
import itertools
import logging
import math
import re
import sys
import warnings
from collections import namedtuple
from contextlib import suppress
from functools import lru_cache, partial
from keyword import iskeyword
from typing import Dict, List, Set, Union
import attr
import pycodestyle
@attr.s
class BugBearVisitor(ast.NodeVisitor):
    filename = attr.ib()
    lines = attr.ib()
    b008_extend_immutable_calls = attr.ib(default=attr.Factory(set))
    b902_classmethod_decorators = attr.ib(default=attr.Factory(set))
    node_window = attr.ib(default=attr.Factory(list))
    errors = attr.ib(default=attr.Factory(list))
    futures = attr.ib(default=attr.Factory(set))
    contexts = attr.ib(default=attr.Factory(list))
    NODE_WINDOW_SIZE = 4
    _b023_seen = attr.ib(factory=set, init=False)
    _b005_imports = attr.ib(factory=set, init=False)
    if False:

        def __getattr__(self, name):
            print(name)
            return self.__getattribute__(name)

    @property
    def node_stack(self):
        if len(self.contexts) == 0:
            return []
        context, stack = self.contexts[-1]
        return stack

    def in_class_init(self) -> bool:
        return len(self.contexts) >= 2 and isinstance(self.contexts[-2].node, ast.ClassDef) and isinstance(self.contexts[-1].node, ast.FunctionDef) and (self.contexts[-1].node.name == '__init__')

    def visit_Return(self, node: ast.Return) -> None:
        if self.in_class_init():
            if node.value is not None:
                self.errors.append(B037(node.lineno, node.col_offset))
        self.generic_visit(node)

    def visit_Yield(self, node: ast.Yield) -> None:
        if self.in_class_init():
            self.errors.append(B037(node.lineno, node.col_offset))
        self.generic_visit(node)

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        if self.in_class_init():
            self.errors.append(B037(node.lineno, node.col_offset))
        self.generic_visit(node)

    def visit(self, node):
        is_contextful = isinstance(node, CONTEXTFUL_NODES)
        if is_contextful:
            context = Context(node, [])
            self.contexts.append(context)
        self.node_stack.append(node)
        self.node_window.append(node)
        self.node_window = self.node_window[-self.NODE_WINDOW_SIZE:]
        super().visit(node)
        self.node_stack.pop()
        if is_contextful:
            self.contexts.pop()
        self.check_for_b018(node)

    def visit_ExceptHandler(self, node):
        if node.type is None:
            self.errors.append(B001(node.lineno, node.col_offset))
            self.generic_visit(node)
            return
        handlers = _flatten_excepthandler(node.type)
        names = []
        bad_handlers = []
        ignored_handlers = []
        for handler in handlers:
            if isinstance(handler, (ast.Name, ast.Attribute)):
                name = _to_name_str(handler)
                if name is None:
                    ignored_handlers.append(handler)
                else:
                    names.append(name)
            elif isinstance(handler, (ast.Call, ast.Starred)):
                ignored_handlers.append(handler)
            else:
                bad_handlers.append(handler)
        if bad_handlers:
            self.errors.append(B030(node.lineno, node.col_offset))
        if len(names) == 0 and (not bad_handlers) and (not ignored_handlers):
            self.errors.append(B029(node.lineno, node.col_offset))
        elif len(names) == 1 and (not bad_handlers) and (not ignored_handlers) and isinstance(node.type, ast.Tuple):
            self.errors.append(B013(node.lineno, node.col_offset, vars=names))
        else:
            maybe_error = _check_redundant_excepthandlers(names, node)
            if maybe_error is not None:
                self.errors.append(maybe_error)
        if 'BaseException' in names and (not ExceptBaseExceptionVisitor(node).re_raised()):
            self.errors.append(B036(node.lineno, node.col_offset))
        self.generic_visit(node)

    def visit_UAdd(self, node):
        trailing_nodes = list(map(type, self.node_window[-4:]))
        if trailing_nodes == [ast.UnaryOp, ast.UAdd, ast.UnaryOp, ast.UAdd]:
            originator = self.node_window[-4]
            self.errors.append(B002(originator.lineno, originator.col_offset))
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            self.check_for_b005(node)
        else:
            with suppress(AttributeError, IndexError):
                if node.func.id in ('getattr', 'hasattr') and node.args[1].value == '__call__':
                    self.errors.append(B004(node.lineno, node.col_offset))
                if node.func.id == 'getattr' and len(node.args) == 2 and _is_identifier(node.args[1]) and (not iskeyword(node.args[1].value)):
                    self.errors.append(B009(node.lineno, node.col_offset))
                elif not any((isinstance(n, ast.Lambda) for n in self.node_stack)) and node.func.id == 'setattr' and (len(node.args) == 3) and _is_identifier(node.args[1]) and (not iskeyword(node.args[1].value)):
                    self.errors.append(B010(node.lineno, node.col_offset))
        self.check_for_b026(node)
        self.check_for_b028(node)
        self.check_for_b034(node)
        self.check_for_b905(node)
        self.generic_visit(node)

    def visit_Module(self, node):
        self.generic_visit(node)

    def visit_Assign(self, node):
        if len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name):
                if (t.value.id, t.attr) == ('os', 'environ'):
                    self.errors.append(B003(node.lineno, node.col_offset))
        self.generic_visit(node)

    def visit_For(self, node):
        self.check_for_b007(node)
        self.check_for_b020(node)
        self.check_for_b023(node)
        self.check_for_b031(node)
        self.check_for_b909(node)
        self.generic_visit(node)

    def visit_AsyncFor(self, node):
        self.check_for_b023(node)
        self.generic_visit(node)

    def visit_While(self, node):
        self.check_for_b023(node)
        self.generic_visit(node)

    def visit_ListComp(self, node):
        self.check_for_b023(node)
        self.generic_visit(node)

    def visit_SetComp(self, node):
        self.check_for_b023(node)
        self.generic_visit(node)

    def visit_DictComp(self, node):
        self.check_for_b023(node)
        self.check_for_b035(node)
        self.generic_visit(node)

    def visit_GeneratorExp(self, node):
        self.check_for_b023(node)
        self.generic_visit(node)

    def visit_Assert(self, node):
        self.check_for_b011(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.check_for_b902(node)
        self.check_for_b006_and_b008(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.check_for_b901(node)
        self.check_for_b902(node)
        self.check_for_b006_and_b008(node)
        self.check_for_b019(node)
        self.check_for_b021(node)
        self.check_for_b906(node)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        self.check_for_b903(node)
        self.check_for_b021(node)
        self.check_for_b024_and_b027(node)
        self.generic_visit(node)

    def visit_Try(self, node):
        self.check_for_b012(node)
        self.check_for_b025(node)
        self.generic_visit(node)

    def visit_Compare(self, node):
        self.check_for_b015(node)
        self.generic_visit(node)

    def visit_Raise(self, node):
        self.check_for_b016(node)
        self.check_for_b904(node)
        self.generic_visit(node)

    def visit_With(self, node):
        self.check_for_b017(node)
        self.check_for_b022(node)
        self.check_for_b908(node)
        self.generic_visit(node)

    def visit_JoinedStr(self, node):
        self.check_for_b907(node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self.check_for_b032(node)
        self.generic_visit(node)

    def visit_Import(self, node):
        self.check_for_b005(node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        self.visit_Import(node)

    def visit_Set(self, node):
        self.check_for_b033(node)
        self.generic_visit(node)

    def check_for_b005(self, node):
        if isinstance(node, ast.Import):
            for name in node.names:
                self._b005_imports.add(name.asname or name.name)
        elif isinstance(node, ast.ImportFrom):
            for name in node.names:
                self._b005_imports.add(f'{node.module}.{name.name or name.asname}')
        elif isinstance(node, ast.Call):
            if node.func.attr not in B005.methods:
                return
            if isinstance(node.func.value, ast.Name) and node.func.value.id in self._b005_imports:
                return
            if len(node.args) != 1 or not isinstance(node.args[0], ast.Constant) or (not isinstance(node.args[0].value, str)):
                return
            call_path = '.'.join(compose_call_path(node.func.value))
            if call_path in B005.valid_paths:
                return
            value = node.args[0].value
            if len(value) == 1:
                return
            if len(value) == len(set(value)):
                return
            self.errors.append(B005(node.lineno, node.col_offset))

    def check_for_b006_and_b008(self, node):
        visitor = FuntionDefDefaultsVisitor(self.b008_extend_immutable_calls)
        visitor.visit(node.args.defaults + node.args.kw_defaults)
        self.errors.extend(visitor.errors)

    def check_for_b007(self, node):
        targets = NameFinder()
        targets.visit(node.target)
        ctrl_names = set(filter(lambda s: not s.startswith('_'), targets.names))
        body = NameFinder()
        for expr in node.body:
            body.visit(expr)
        used_names = set(body.names)
        for name in sorted(ctrl_names - used_names):
            n = targets.names[name][0]
            self.errors.append(B007(n.lineno, n.col_offset, vars=(name,)))

    def check_for_b011(self, node):
        if isinstance(node.test, ast.Constant) and node.test.value is False:
            self.errors.append(B011(node.lineno, node.col_offset))

    def check_for_b012(self, node):

        def _loop(node, bad_node_types):
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                return
            if isinstance(node, (ast.While, ast.For)):
                bad_node_types = (ast.Return,)
            elif isinstance(node, bad_node_types):
                self.errors.append(B012(node.lineno, node.col_offset))
            for child in ast.iter_child_nodes(node):
                _loop(child, bad_node_types)
        for child in node.finalbody:
            _loop(child, (ast.Return, ast.Continue, ast.Break))

    def check_for_b015(self, node):
        if isinstance(self.node_stack[-2], ast.Expr):
            self.errors.append(B015(node.lineno, node.col_offset))

    def check_for_b016(self, node):
        if isinstance(node.exc, ast.JoinedStr) or (isinstance(node.exc, ast.Constant) and (isinstance(node.exc.value, (int, float, complex, str, bool)) or node.exc.value is None)):
            self.errors.append(B016(node.lineno, node.col_offset))

    def check_for_b017(self, node):
        """Checks for use of the evil syntax 'with assertRaises(Exception):'
        or 'with pytest.raises(Exception)'.

        This form of assertRaises will catch everything that subclasses
        Exception, which happens to be the vast majority of Python internal
        errors, including the ones raised when a non-existing method/function
        is called, or a function is called with an invalid dictionary key
        lookup.
        """
        item = node.items[0]
        item_context = item.context_expr
        if hasattr(item_context, 'func') and (isinstance(item_context.func, ast.Attribute) and (item_context.func.attr == 'assertRaises' or (item_context.func.attr == 'raises' and isinstance(item_context.func.value, ast.Name) and (item_context.func.value.id == 'pytest') and ('match' not in (kwd.arg for kwd in item_context.keywords)))) or (isinstance(item_context.func, ast.Name) and item_context.func.id == 'raises' and isinstance(item_context.func.ctx, ast.Load) and ('pytest.raises' in self._b005_imports) and ('match' not in (kwd.arg for kwd in item_context.keywords)))) and (len(item_context.args) == 1) and isinstance(item_context.args[0], ast.Name) and (item_context.args[0].id in {'Exception', 'BaseException'}) and (not item.optional_vars):
            self.errors.append(B017(node.lineno, node.col_offset))

    def check_for_b019(self, node):
        if len(node.decorator_list) == 0 or len(self.contexts) < 2 or (not isinstance(self.contexts[-2].node, ast.ClassDef)):
            return
        resolved_decorators = ('.'.join(compose_call_path(decorator)) for decorator in node.decorator_list)
        for idx, decorator in enumerate(resolved_decorators):
            if decorator in {'classmethod', 'staticmethod'}:
                return
            if decorator in B019.caches:
                self.errors.append(B019(node.decorator_list[idx].lineno, node.decorator_list[idx].col_offset))
                return

    def check_for_b020(self, node):
        targets = NameFinder()
        targets.visit(node.target)
        ctrl_names = set(targets.names)
        iterset = B020NameFinder()
        iterset.visit(node.iter)
        iterset_names = set(iterset.names)
        for name in sorted(ctrl_names):
            if name in iterset_names:
                n = targets.names[name][0]
                self.errors.append(B020(n.lineno, n.col_offset, vars=(name,)))

    def check_for_b023(self, loop_node):
        """Check that functions (including lambdas) do not use loop variables.

        https://docs.python-guide.org/writing/gotchas/#late-binding-closures from
        functions - usually but not always lambdas - defined inside a loop are a
        classic source of bugs.

        For each use of a variable inside a function defined inside a loop, we
        emit a warning if that variable is reassigned on each loop iteration
        (outside the function).  This includes but is not limited to explicit
        loop variables like the `x` in `for x in range(3):`.
        """
        safe_functions = []
        suspicious_variables = []
        for node in ast.walk(loop_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in ('filter', 'reduce', 'map') or (isinstance(node.func, ast.Attribute) and node.func.attr == 'reduce' and isinstance(node.func.value, ast.Name) and (node.func.value.id == 'functools')):
                    for arg in node.args:
                        if isinstance(arg, FUNCTION_NODES):
                            safe_functions.append(arg)
                for keyword in node.keywords:
                    if keyword.arg == 'key' and isinstance(keyword.value, FUNCTION_NODES):
                        safe_functions.append(keyword.value)
            if isinstance(node, ast.Return):
                if isinstance(node.value, FUNCTION_NODES):
                    safe_functions.append(node.value)
            if isinstance(node, FUNCTION_NODES) and node not in safe_functions:
                argnames = {arg.arg for arg in ast.walk(node.args) if isinstance(arg, ast.arg)}
                if isinstance(node, ast.Lambda):
                    body_nodes = ast.walk(node.body)
                else:
                    body_nodes = itertools.chain.from_iterable(map(ast.walk, node.body))
                errors = []
                for name in body_nodes:
                    if isinstance(name, ast.Name) and name.id not in argnames:
                        if isinstance(name.ctx, ast.Load):
                            errors.append(B023(name.lineno, name.col_offset, vars=(name.id,)))
                        elif isinstance(name.ctx, ast.Store):
                            argnames.add(name.id)
                for err in errors:
                    if err.vars[0] not in argnames and err not in self._b023_seen:
                        self._b023_seen.add(err)
                        suspicious_variables.append(err)
        if suspicious_variables:
            reassigned_in_loop = set(self._get_assigned_names(loop_node))
        for err in sorted(suspicious_variables):
            if reassigned_in_loop.issuperset(err.vars):
                self.errors.append(err)

    def check_for_b024_and_b027(self, node: ast.ClassDef):
        """Check for inheritance from abstract classes in abc and lack of
        any methods decorated with abstract*"""

        def is_abc_class(value, name='ABC'):
            if isinstance(value, ast.keyword):
                return value.arg == 'metaclass' and is_abc_class(value.value, 'ABCMeta')
            return isinstance(value, ast.Name) and value.id == name or (isinstance(value, ast.Attribute) and value.attr == name and isinstance(value.value, ast.Name) and (value.value.id == 'abc'))

        def is_abstract_decorator(expr):
            return isinstance(expr, ast.Name) and expr.id[:8] == 'abstract' or (isinstance(expr, ast.Attribute) and expr.attr[:8] == 'abstract')

        def is_overload(expr):
            return isinstance(expr, ast.Name) and expr.id == 'overload' or (isinstance(expr, ast.Attribute) and expr.attr == 'overload')

        def empty_body(body) -> bool:

            def is_str_or_ellipsis(node):
                return isinstance(node, ast.Constant) and (node.value is Ellipsis or isinstance(node.value, str))
            return all((isinstance(stmt, ast.Pass) or (isinstance(stmt, ast.Expr) and is_str_or_ellipsis(stmt.value)) for stmt in body))
        if len(node.bases) + len(node.keywords) > 1:
            return
        if not any(map(is_abc_class, (*node.bases, *node.keywords))):
            return
        has_method = False
        has_abstract_method = False
        for stmt in node.body:
            if isinstance(stmt, (ast.AnnAssign, ast.Assign)):
                has_abstract_method = True
                continue
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            has_method = True
            has_abstract_decorator = any(map(is_abstract_decorator, stmt.decorator_list))
            has_abstract_method |= has_abstract_decorator
            if not has_abstract_decorator and empty_body(stmt.body) and (not any(map(is_overload, stmt.decorator_list))):
                self.errors.append(B027(stmt.lineno, stmt.col_offset, vars=(stmt.name,)))
        if has_method and (not has_abstract_method):
            self.errors.append(B024(node.lineno, node.col_offset, vars=(node.name,)))

    def check_for_b026(self, call: ast.Call):
        if not call.keywords:
            return
        starreds = [arg for arg in call.args if isinstance(arg, ast.Starred)]
        if not starreds:
            return
        first_keyword = call.keywords[0].value
        for starred in starreds:
            if (starred.lineno, starred.col_offset) > (first_keyword.lineno, first_keyword.col_offset):
                self.errors.append(B026(starred.lineno, starred.col_offset))

    def check_for_b031(self, loop_node):
        """Check that `itertools.groupby` isn't iterated over more than once.

        We emit a warning when the generator returned by `groupby()` is used
        more than once inside a loop body or when it's used in a nested loop.
        """
        if isinstance(loop_node.iter, ast.Call):
            node = loop_node.iter
            if isinstance(node.func, ast.Name) and node.func.id in ('groupby',) or (isinstance(node.func, ast.Attribute) and node.func.attr == 'groupby' and isinstance(node.func.value, ast.Name) and (node.func.value.id == 'itertools')):
                if isinstance(loop_node.target, ast.Tuple) and isinstance(loop_node.target.elts[1], ast.Name):
                    group_name = loop_node.target.elts[1].id
                else:
                    return
                num_usages = 0
                for node in walk_list(loop_node.body):
                    if isinstance(node, ast.For):
                        for nested_node in walk_list(node.body):
                            assert nested_node != node
                            if isinstance(nested_node, ast.Name) and nested_node.id == group_name:
                                self.errors.append(B031(nested_node.lineno, nested_node.col_offset, vars=(nested_node.id,)))
                    if isinstance(node, ast.Name) and node.id == group_name:
                        num_usages += 1
                        if num_usages > 1:
                            self.errors.append(B031(node.lineno, node.col_offset, vars=(node.id,)))

    def _get_names_from_tuple(self, node: ast.Tuple):
        for dim in node.elts:
            if isinstance(dim, ast.Name):
                yield dim.id
            elif isinstance(dim, ast.Tuple):
                yield from self._get_names_from_tuple(dim)

    def _get_dict_comp_loop_and_named_expr_var_names(self, node: ast.DictComp):
        finder = NamedExprFinder()
        for gen in node.generators:
            if isinstance(gen.target, ast.Name):
                yield gen.target.id
            elif isinstance(gen.target, ast.Tuple):
                yield from self._get_names_from_tuple(gen.target)
            finder.visit(gen.ifs)
        yield from finder.names.keys()

    def check_for_b035(self, node: ast.DictComp):
        """Check that a static key isn't used in a dict comprehension.

        Emit a warning if a likely unchanging key is used - either a constant,
        or a variable that isn't coming from the generator expression.
        """
        if isinstance(node.key, ast.Constant):
            self.errors.append(B035(node.key.lineno, node.key.col_offset, vars=(node.key.value,)))
        elif isinstance(node.key, ast.Name):
            if node.key.id not in self._get_dict_comp_loop_and_named_expr_var_names(node):
                self.errors.append(B035(node.key.lineno, node.key.col_offset, vars=(node.key.id,)))

    def _get_assigned_names(self, loop_node):
        loop_targets = (ast.For, ast.AsyncFor, ast.comprehension)
        for node in children_in_scope(loop_node):
            if isinstance(node, ast.Assign):
                for child in node.targets:
                    yield from names_from_assignments(child)
            if isinstance(node, loop_targets + (ast.AnnAssign, ast.AugAssign)):
                yield from names_from_assignments(node.target)

    def check_for_b904(self, node):
        """Checks `raise` without `from` inside an `except` clause.

        In these cases, you should use explicit exception chaining from the
        earlier error, or suppress it with `raise ... from None`.  See
        https://docs.python.org/3/tutorial/errors.html#exception-chaining
        """
        if node.cause is None and node.exc is not None and (not (isinstance(node.exc, ast.Name) and node.exc.id.islower())) and any((isinstance(n, ast.ExceptHandler) for n in self.node_stack)):
            self.errors.append(B904(node.lineno, node.col_offset))

    def walk_function_body(self, node):

        def _loop(parent, node):
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                return
            yield (parent, node)
            for child in ast.iter_child_nodes(node):
                yield from _loop(node, child)
        for child in node.body:
            yield from _loop(node, child)

    def check_for_b901(self, node):
        if node.name == '__await__':
            return
        has_yield = False
        return_node = None
        for parent, x in self.walk_function_body(node):
            if isinstance(parent, ast.Expr) and isinstance(x, (ast.Yield, ast.YieldFrom)):
                has_yield = True
            if isinstance(x, ast.Return) and x.value is not None:
                return_node = x
            if has_yield and return_node is not None:
                self.errors.append(B901(return_node.lineno, return_node.col_offset))
                break

    @classmethod
    def find_decorator_name(cls, d):
        if isinstance(d, ast.Name):
            return d.id
        elif isinstance(d, ast.Attribute):
            return d.attr
        elif isinstance(d, ast.Call):
            return cls.find_decorator_name(d.func)

    def check_for_b902(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> None:

        def is_classmethod(decorators: Set[str]) -> bool:
            return any((name in decorators for name in self.b902_classmethod_decorators)) or node.name in B902.implicit_classmethods
        if len(self.contexts) < 2 or not isinstance(self.contexts[-2].node, ast.ClassDef):
            return
        cls = self.contexts[-2].node
        decorators: set[str] = {self.find_decorator_name(d) for d in node.decorator_list}
        if 'staticmethod' in decorators:
            return
        bases = {b.id for b in cls.bases if isinstance(b, ast.Name)}
        if any((basetype in bases for basetype in ('type', 'ABCMeta', 'EnumMeta'))):
            if is_classmethod(decorators):
                expected_first_args = B902.metacls
                kind = 'metaclass class'
            else:
                expected_first_args = B902.cls
                kind = 'metaclass instance'
        elif is_classmethod(decorators):
            expected_first_args = B902.cls
            kind = 'class'
        else:
            expected_first_args = B902.self
            kind = 'instance'
        args = getattr(node.args, 'posonlyargs', []) + node.args.args
        vararg = node.args.vararg
        kwarg = node.args.kwarg
        kwonlyargs = node.args.kwonlyargs
        if args:
            actual_first_arg = args[0].arg
            lineno = args[0].lineno
            col = args[0].col_offset
        elif vararg:
            actual_first_arg = '*' + vararg.arg
            lineno = vararg.lineno
            col = vararg.col_offset
        elif kwarg:
            actual_first_arg = '**' + kwarg.arg
            lineno = kwarg.lineno
            col = kwarg.col_offset
        elif kwonlyargs:
            actual_first_arg = '*, ' + kwonlyargs[0].arg
            lineno = kwonlyargs[0].lineno
            col = kwonlyargs[0].col_offset
        else:
            actual_first_arg = '(none)'
            lineno = node.lineno
            col = node.col_offset
        if actual_first_arg not in expected_first_args:
            if not actual_first_arg.startswith(('(', '*')):
                actual_first_arg = repr(actual_first_arg)
            self.errors.append(B902(lineno, col, vars=(actual_first_arg, kind, expected_first_args[0])))

    def check_for_b903(self, node):
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            body = body[1:]
        if len(body) != 1 or not isinstance(body[0], ast.FunctionDef) or body[0].name != '__init__':
            return
        for stmt in body[0].body:
            if not isinstance(stmt, ast.Assign):
                return
            targets = stmt.targets
            if len(targets) > 1 or not isinstance(targets[0], ast.Attribute):
                return
            if not isinstance(stmt.value, ast.Name):
                return
        self.errors.append(B903(node.lineno, node.col_offset))

    def check_for_b018(self, node):
        if not isinstance(node, ast.Expr):
            return
        if isinstance(node.value, (ast.List, ast.Set, ast.Dict, ast.Tuple)) or (isinstance(node.value, ast.Constant) and (isinstance(node.value.value, (int, float, complex, bytes, bool)) or node.value.value is None)):
            self.errors.append(B018(node.lineno, node.col_offset, vars=(node.value.__class__.__name__,)))

    def check_for_b021(self, node):
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.JoinedStr):
            self.errors.append(B021(node.body[0].value.lineno, node.body[0].value.col_offset))

    def check_for_b022(self, node):
        item = node.items[0]
        item_context = item.context_expr
        if hasattr(item_context, 'func') and hasattr(item_context.func, 'value') and hasattr(item_context.func.value, 'id') and (item_context.func.value.id == 'contextlib') and hasattr(item_context.func, 'attr') and (item_context.func.attr == 'suppress') and (len(item_context.args) == 0):
            self.errors.append(B022(node.lineno, node.col_offset))

    @staticmethod
    def _is_assertRaises_like(node: ast.withitem) -> bool:
        if not (isinstance(node, ast.withitem) and isinstance(node.context_expr, ast.Call) and isinstance(node.context_expr.func, (ast.Attribute, ast.Name))):
            return False
        if isinstance(node.context_expr.func, ast.Name):
            return node.context_expr.func.id in B908_pytest_functions
        elif isinstance(node.context_expr.func, ast.Attribute) and isinstance(node.context_expr.func.value, ast.Name):
            return node.context_expr.func.value.id == 'pytest' and node.context_expr.func.attr in B908_pytest_functions or (node.context_expr.func.value.id == 'self' and node.context_expr.func.attr in B908_unittest_methods)
        else:
            return False

    def check_for_b908(self, node: ast.With):
        if len(node.body) < 2:
            return
        for node_item in node.items:
            if self._is_assertRaises_like(node_item):
                self.errors.append(B908(node.lineno, node.col_offset))

    def check_for_b025(self, node):
        seen = []
        for handler in node.handlers:
            if isinstance(handler.type, (ast.Name, ast.Attribute)):
                name = '.'.join(compose_call_path(handler.type))
                seen.append(name)
            elif isinstance(handler.type, ast.Tuple):
                uniques = set()
                for entry in handler.type.elts:
                    name = '.'.join(compose_call_path(entry))
                    uniques.add(name)
                seen.extend(uniques)
        duplicates = sorted({x for x in seen if seen.count(x) > 1})
        for duplicate in duplicates:
            self.errors.append(B025(node.lineno, node.col_offset, vars=(duplicate,)))

    @staticmethod
    def _is_infinite_iterator(node: ast.expr) -> bool:
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and (node.func.value.id == 'itertools')):
            return False
        if node.func.attr in {'cycle', 'count'}:
            return True
        elif node.func.attr == 'repeat':
            if len(node.args) == 1 and len(node.keywords) == 0:
                return True
            if len(node.args) == 2 and isinstance(node.args[1], ast.Constant) and (node.args[1].value is None):
                return True
            for kw in node.keywords:
                if kw.arg == 'times' and isinstance(kw.value, ast.Constant) and (kw.value.value is None):
                    return True
        return False

    def check_for_b905(self, node):
        if not (isinstance(node.func, ast.Name) and node.func.id == 'zip'):
            return
        for arg in node.args:
            if self._is_infinite_iterator(arg):
                return
        if not any((kw.arg == 'strict' for kw in node.keywords)):
            self.errors.append(B905(node.lineno, node.col_offset))

    def check_for_b906(self, node: ast.FunctionDef):
        if not node.name.startswith('visit_'):
            return
        class_name = node.name[len('visit_'):]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            class_type = getattr(ast, class_name, None)
        if class_type is None or not getattr(class_type, '_fields', None) or class_type.__name__ in ('alias', 'Constant', 'Global', 'MatchSingleton', 'MatchStar', 'Nonlocal', 'TypeIgnore', 'Bytes', 'Num', 'Str'):
            return
        for n in itertools.chain.from_iterable((ast.walk(nn) for nn in node.body)):
            if isinstance(n, ast.Call) and (isinstance(n.func, ast.Attribute) and 'visit' in n.func.attr or (isinstance(n.func, ast.Name) and 'visit' in n.func.id)):
                break
        else:
            self.errors.append(B906(node.lineno, node.col_offset))

    def check_for_b907(self, node: ast.JoinedStr):

        def myunparse(node: ast.AST) -> str:
            if sys.version_info >= (3, 9):
                return ast.unparse(node)
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Attribute):
                return myunparse(node.value) + '.' + node.attr
            if isinstance(node, ast.Constant):
                return repr(node.value)
            if isinstance(node, ast.Call):
                return myunparse(node.func) + '()'
            return type(node).__name__
        quote_marks = '\'"'
        current_mark = None
        variable = None
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                if not value.value:
                    continue
                if current_mark is not None and variable is not None and (value.value[0] == current_mark):
                    self.errors.append(B907(variable.lineno, variable.col_offset, vars=(myunparse(variable.value),)))
                    current_mark = variable = None
                    if len(value.value) == 1:
                        continue
                if value.value[-1] in quote_marks:
                    current_mark = value.value[-1]
                    variable = None
                    continue
            if current_mark is not None and variable is None and isinstance(value, ast.FormattedValue) and (value.conversion != ord('r')):
                if isinstance(value.format_spec, ast.JoinedStr) and value.format_spec.values:
                    if len(value.format_spec.values) > 1 or not isinstance(value.format_spec.values[0], ast.Constant):
                        current_mark = variable = None
                        continue
                    format_specifier = value.format_spec.values[0].value
                    if len(format_specifier) > 1 and format_specifier[1] in '<>=^':
                        format_specifier = format_specifier[1:]
                    format_specifier = re.sub('\\.\\d*', '', format_specifier)
                    invalid_characters = ''.join(('=', '+- ', '0123456789', 'z', '#', '_,', 'bcdeEfFgGnoxX%'))
                    if set(format_specifier).intersection(invalid_characters):
                        current_mark = variable = None
                        continue
                variable = value
                continue
            current_mark = variable = None

    def check_for_b028(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'warn' and isinstance(node.func.value, ast.Name) and (node.func.value.id == 'warnings') and (not any((kw.arg == 'stacklevel' for kw in node.keywords))) and (len(node.args) < 3):
            self.errors.append(B028(node.lineno, node.col_offset))

    def check_for_b032(self, node):
        if node.value is None and hasattr(node.target, 'value') and isinstance(node.target.value, ast.Name) and (isinstance(node.target, ast.Subscript) or (isinstance(node.target, ast.Attribute) and node.target.value.id != 'self')):
            self.errors.append(B032(node.lineno, node.col_offset))

    def check_for_b033(self, node):
        seen = set()
        for elt in node.elts:
            if not isinstance(elt, ast.Constant):
                continue
            if elt.value in seen:
                self.errors.append(B033(elt.lineno, elt.col_offset, vars=(repr(elt.value),)))
            else:
                seen.add(elt.value)

    def check_for_b034(self, node: ast.Call):
        if not isinstance(node.func, ast.Attribute):
            return
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != 're':
            return

        def check(num_args, param_name):
            if len(node.args) > num_args:
                self.errors.append(B034(node.args[num_args].lineno, node.args[num_args].col_offset, vars=(node.func.attr, param_name)))
        if node.func.attr in ('sub', 'subn'):
            check(3, 'count')
        elif node.func.attr == 'split':
            check(2, 'maxsplit')

    def check_for_b909(self, node: ast.For):
        if isinstance(node.iter, ast.Name):
            name = _to_name_str(node.iter)
        elif isinstance(node.iter, ast.Attribute):
            name = _to_name_str(node.iter)
        else:
            return
        checker = B909Checker(name)
        checker.visit(node.body)
        for mutation in checker.mutations:
            self.errors.append(B909(mutation.lineno, mutation.col_offset))