from __future__ import annotations
import ast
import builtins
import sys
import typing
from ast import (
from collections import defaultdict
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast, overload
class TypeguardTransformer(NodeTransformer):

    def __init__(self, target_path: Sequence[str] | None=None, target_lineno: int | None=None) -> None:
        self._target_path = tuple(target_path) if target_path else None
        self._memo = self._module_memo = TransformMemo(None, None, ())
        self.names_used_in_annotations: set[str] = set()
        self.target_node: FunctionDef | AsyncFunctionDef | None = None
        self.target_lineno = target_lineno

    def generic_visit(self, node: AST) -> AST:
        has_non_empty_body_initially = bool(getattr(node, 'body', None))
        initial_type = type(node)
        node = super().generic_visit(node)
        if type(node) is initial_type and has_non_empty_body_initially and hasattr(node, 'body') and (not node.body):
            node.body = [Pass()]
        return node

    @contextmanager
    def _use_memo(self, node: ClassDef | FunctionDef | AsyncFunctionDef) -> Generator[None, Any, None]:
        new_memo = TransformMemo(node, self._memo, self._memo.path + (node.name,))
        old_memo = self._memo
        self._memo = new_memo
        if isinstance(node, (FunctionDef, AsyncFunctionDef)):
            new_memo.should_instrument = self._target_path is None or new_memo.path == self._target_path
            if new_memo.should_instrument:
                detector = GeneratorDetector()
                detector.visit(node)
                return_annotation = deepcopy(node.returns)
                if detector.contains_yields and new_memo.name_matches(return_annotation, *generator_names):
                    if isinstance(return_annotation, Subscript):
                        annotation_slice = return_annotation.slice
                        if isinstance(annotation_slice, Index):
                            annotation_slice = annotation_slice.value
                        if isinstance(annotation_slice, Tuple):
                            items = annotation_slice.elts
                        else:
                            items = [annotation_slice]
                        if len(items) > 0:
                            new_memo.yield_annotation = self._convert_annotation(items[0])
                        if len(items) > 1:
                            new_memo.send_annotation = self._convert_annotation(items[1])
                        if len(items) > 2:
                            new_memo.return_annotation = self._convert_annotation(items[2])
                else:
                    new_memo.return_annotation = self._convert_annotation(return_annotation)
        if isinstance(node, AsyncFunctionDef):
            new_memo.is_async = True
        yield
        self._memo = old_memo

    def _get_import(self, module: str, name: str) -> Name:
        memo = self._memo if self._target_path else self._module_memo
        return memo.get_import(module, name)

    @overload
    def _convert_annotation(self, annotation: None) -> None:
        ...

    @overload
    def _convert_annotation(self, annotation: expr) -> expr:
        ...

    def _convert_annotation(self, annotation: expr | None) -> expr | None:
        if annotation is None:
            return None
        new_annotation = cast(expr, AnnotationTransformer(self).visit(annotation))
        if isinstance(new_annotation, expr):
            new_annotation = ast.copy_location(new_annotation, annotation)
            names = {node.id for node in walk(new_annotation) if isinstance(node, Name)}
            self.names_used_in_annotations.update(names)
        return new_annotation

    def visit_Name(self, node: Name) -> Name:
        self._memo.local_names.add(node.id)
        return node

    def visit_Module(self, node: Module) -> Module:
        self._module_memo = self._memo = TransformMemo(node, None, ())
        self.generic_visit(node)
        self._module_memo.insert_imports(node)
        fix_missing_locations(node)
        return node

    def visit_Import(self, node: Import) -> Import:
        for name in node.names:
            self._memo.local_names.add(name.asname or name.name)
            self._memo.imported_names[name.asname or name.name] = name.name
        return node

    def visit_ImportFrom(self, node: ImportFrom) -> ImportFrom:
        for name in node.names:
            if name.name != '*':
                alias = name.asname or name.name
                self._memo.local_names.add(alias)
                self._memo.imported_names[alias] = f'{node.module}.{name.name}'
        return node

    def visit_ClassDef(self, node: ClassDef) -> ClassDef | None:
        self._memo.local_names.add(node.name)
        if self._target_path is not None and (not self._memo.path) and (node.name != self._target_path[0]):
            return None
        with self._use_memo(node):
            for decorator in node.decorator_list.copy():
                if self._memo.name_matches(decorator, 'typeguard.typechecked'):
                    node.decorator_list.remove(decorator)
                    if isinstance(decorator, Call) and decorator.keywords:
                        self._memo.configuration_overrides.update({kw.arg: kw.value for kw in decorator.keywords if kw.arg})
            self.generic_visit(node)
            return node

    def visit_FunctionDef(self, node: FunctionDef | AsyncFunctionDef) -> FunctionDef | AsyncFunctionDef | None:
        """
        Injects type checks for function arguments, and for a return of None if the
        function is annotated to return something else than Any or None, and the body
        ends without an explicit "return".

        """
        self._memo.local_names.add(node.name)
        if self._target_path is not None and (not self._memo.path) and (node.name != self._target_path[0]):
            return None
        if self._target_path is None:
            for decorator in node.decorator_list:
                if self._memo.name_matches(decorator, *ignore_decorators):
                    return node
        with self._use_memo(node):
            arg_annotations: dict[str, Any] = {}
            if self._target_path is None or self._memo.path == self._target_path:
                if node.decorator_list:
                    first_lineno = node.decorator_list[0].lineno
                else:
                    first_lineno = node.lineno
                for decorator in node.decorator_list.copy():
                    if self._memo.name_matches(decorator, 'typing.overload'):
                        return None
                    elif self._memo.name_matches(decorator, 'typeguard.typechecked'):
                        node.decorator_list.remove(decorator)
                        if isinstance(decorator, Call) and decorator.keywords:
                            self._memo.configuration_overrides = {kw.arg: kw.value for kw in decorator.keywords if kw.arg}
                if self.target_lineno == first_lineno:
                    assert self.target_node is None
                    self.target_node = node
                    if node.decorator_list:
                        self.target_lineno = node.decorator_list[0].lineno
                    else:
                        self.target_lineno = node.lineno
                all_args = node.args.args + node.args.kwonlyargs + node.args.posonlyargs
                for arg in all_args:
                    self._memo.ignored_names.add(arg.arg)
                if node.args.vararg:
                    self._memo.ignored_names.add(node.args.vararg.arg)
                if node.args.kwarg:
                    self._memo.ignored_names.add(node.args.kwarg.arg)
                for arg in all_args:
                    annotation = self._convert_annotation(deepcopy(arg.annotation))
                    if annotation:
                        arg_annotations[arg.arg] = annotation
                if node.args.vararg:
                    annotation_ = self._convert_annotation(node.args.vararg.annotation)
                    if annotation_:
                        if sys.version_info >= (3, 9):
                            container = Name('tuple', ctx=Load())
                        else:
                            container = self._get_import('typing', 'Tuple')
                        subscript_slice: Tuple | Index = Tuple([annotation_, Constant(Ellipsis)], ctx=Load())
                        if sys.version_info < (3, 9):
                            subscript_slice = Index(subscript_slice, ctx=Load())
                        arg_annotations[node.args.vararg.arg] = Subscript(container, subscript_slice, ctx=Load())
                if node.args.kwarg:
                    annotation_ = self._convert_annotation(node.args.kwarg.annotation)
                    if annotation_:
                        if sys.version_info >= (3, 9):
                            container = Name('dict', ctx=Load())
                        else:
                            container = self._get_import('typing', 'Dict')
                        subscript_slice = Tuple([Name('str', ctx=Load()), annotation_], ctx=Load())
                        if sys.version_info < (3, 9):
                            subscript_slice = Index(subscript_slice, ctx=Load())
                        arg_annotations[node.args.kwarg.arg] = Subscript(container, subscript_slice, ctx=Load())
                if arg_annotations:
                    self._memo.variable_annotations.update(arg_annotations)
            self.generic_visit(node)
            if arg_annotations:
                annotations_dict = Dict(keys=[Constant(key) for key in arg_annotations.keys()], values=[Tuple([Name(key, ctx=Load()), annotation], ctx=Load()) for key, annotation in arg_annotations.items()])
                func_name = self._get_import('typeguard._functions', 'check_argument_types')
                args = [self._memo.joined_path, annotations_dict, self._memo.get_memo_name()]
                node.body.insert(self._memo.code_inject_index, Expr(Call(func_name, args, [])))
            if self._memo.return_annotation and (not self._memo.is_async or not self._memo.has_yield_expressions) and (not isinstance(node.body[-1], Return)) and (not isinstance(self._memo.return_annotation, Constant) or self._memo.return_annotation.value is not None):
                func_name = self._get_import('typeguard._functions', 'check_return_type')
                return_node = Return(Call(func_name, [self._memo.joined_path, Constant(None), self._memo.return_annotation, self._memo.get_memo_name()], []))
                if isinstance(node.body[-1], Pass):
                    copy_location(return_node, node.body[-1])
                    del node.body[-1]
                node.body.append(return_node)
            if self._memo.memo_var_name:
                memo_kwargs: dict[str, Any] = {}
                if self._memo.parent and isinstance(self._memo.parent.node, ClassDef):
                    for decorator in node.decorator_list:
                        if isinstance(decorator, Name) and decorator.id == 'staticmethod':
                            break
                        elif isinstance(decorator, Name) and decorator.id == 'classmethod':
                            memo_kwargs['self_type'] = Name(id=node.args.args[0].arg, ctx=Load())
                            break
                    else:
                        if node.args.args:
                            if node.name == '__new__':
                                memo_kwargs['self_type'] = Name(id=node.args.args[0].arg, ctx=Load())
                            else:
                                memo_kwargs['self_type'] = Attribute(Name(id=node.args.args[0].arg, ctx=Load()), '__class__', ctx=Load())
                names: list[str] = [node.name]
                memo = self._memo.parent
                while memo:
                    if isinstance(memo.node, (FunctionDef, AsyncFunctionDef)):
                        del names[:-1]
                        break
                    elif not isinstance(memo.node, ClassDef):
                        break
                    names.insert(0, memo.node.name)
                    memo = memo.parent
                config_keywords = self._memo.get_config_keywords()
                if config_keywords:
                    memo_kwargs['config'] = Call(self._get_import('dataclasses', 'replace'), [self._get_import('typeguard._config', 'global_config')], config_keywords)
                self._memo.memo_var_name.id = self._memo.get_unused_name('memo')
                memo_store_name = Name(id=self._memo.memo_var_name.id, ctx=Store())
                globals_call = Call(Name(id='globals', ctx=Load()), [], [])
                locals_call = Call(Name(id='locals', ctx=Load()), [], [])
                memo_expr = Call(self._get_import('typeguard', 'TypeCheckMemo'), [globals_call, locals_call], [keyword(key, value) for key, value in memo_kwargs.items()])
                node.body.insert(self._memo.code_inject_index, Assign([memo_store_name], memo_expr))
                self._memo.insert_imports(node)
                if isinstance(node, FunctionDef) and node.args and (self._memo.parent is not None) and isinstance(self._memo.parent.node, ClassDef) and (node.name == '__new__'):
                    first_args_expr = Name(node.args.args[0].arg, ctx=Load())
                    cls_name = Name(self._memo.parent.node.name, ctx=Store())
                    node.body.insert(self._memo.code_inject_index, Assign([cls_name], first_args_expr))
                if isinstance(node.body[-1], Pass):
                    del node.body[-1]
        return node

    def visit_AsyncFunctionDef(self, node: AsyncFunctionDef) -> FunctionDef | AsyncFunctionDef | None:
        return self.visit_FunctionDef(node)

    def visit_Return(self, node: Return) -> Return:
        """This injects type checks into "return" statements."""
        self.generic_visit(node)
        if self._memo.return_annotation and self._memo.should_instrument and (not self._memo.is_ignored_name(self._memo.return_annotation)):
            func_name = self._get_import('typeguard._functions', 'check_return_type')
            old_node = node
            retval = old_node.value or Constant(None)
            node = Return(Call(func_name, [self._memo.joined_path, retval, self._memo.return_annotation, self._memo.get_memo_name()], []))
            copy_location(node, old_node)
        return node

    def visit_Yield(self, node: Yield) -> Yield | Call:
        """
        This injects type checks into "yield" expressions, checking both the yielded
        value and the value sent back to the generator, when appropriate.

        """
        self._memo.has_yield_expressions = True
        self.generic_visit(node)
        if self._memo.yield_annotation and self._memo.should_instrument and (not self._memo.is_ignored_name(self._memo.yield_annotation)):
            func_name = self._get_import('typeguard._functions', 'check_yield_type')
            yieldval = node.value or Constant(None)
            node.value = Call(func_name, [self._memo.joined_path, yieldval, self._memo.yield_annotation, self._memo.get_memo_name()], [])
        if self._memo.send_annotation and self._memo.should_instrument and (not self._memo.is_ignored_name(self._memo.send_annotation)):
            func_name = self._get_import('typeguard._functions', 'check_send_type')
            old_node = node
            call_node = Call(func_name, [self._memo.joined_path, old_node, self._memo.send_annotation, self._memo.get_memo_name()], [])
            copy_location(call_node, old_node)
            return call_node
        return node

    def visit_AnnAssign(self, node: AnnAssign) -> Any:
        """
        This injects a type check into a local variable annotation-assignment within a
        function body.

        """
        self.generic_visit(node)
        if isinstance(self._memo.node, (FunctionDef, AsyncFunctionDef)) and node.annotation and isinstance(node.target, Name):
            self._memo.ignored_names.add(node.target.id)
            annotation = self._convert_annotation(deepcopy(node.annotation))
            if annotation:
                self._memo.variable_annotations[node.target.id] = annotation
                if node.value:
                    func_name = self._get_import('typeguard._functions', 'check_variable_assignment')
                    node.value = Call(func_name, [node.value, Constant(node.target.id), annotation, self._memo.get_memo_name()], [])
        return node

    def visit_Assign(self, node: Assign) -> Any:
        """
        This injects a type check into a local variable assignment within a function
        body. The variable must have been annotated earlier in the function body.

        """
        self.generic_visit(node)
        if isinstance(self._memo.node, (FunctionDef, AsyncFunctionDef)):
            targets: list[dict[Constant, expr | None]] = []
            check_required = False
            for target in node.targets:
                elts: Sequence[expr]
                if isinstance(target, Name):
                    elts = [target]
                elif isinstance(target, Tuple):
                    elts = target.elts
                else:
                    continue
                annotations_: dict[Constant, expr | None] = {}
                for exp in elts:
                    prefix = ''
                    if isinstance(exp, Starred):
                        exp = exp.value
                        prefix = '*'
                    if isinstance(exp, Name):
                        self._memo.ignored_names.add(exp.id)
                        name = prefix + exp.id
                        annotation = self._memo.variable_annotations.get(exp.id)
                        if annotation:
                            annotations_[Constant(name)] = annotation
                            check_required = True
                        else:
                            annotations_[Constant(name)] = None
                targets.append(annotations_)
            if check_required:
                for item in targets:
                    for key, expression in item.items():
                        if expression is None:
                            item[key] = self._get_import('typing', 'Any')
                if len(targets) == 1 and len(targets[0]) == 1:
                    func_name = self._get_import('typeguard._functions', 'check_variable_assignment')
                    target_varname = next(iter(targets[0]))
                    node.value = Call(func_name, [node.value, target_varname, targets[0][target_varname], self._memo.get_memo_name()], [])
                elif targets:
                    func_name = self._get_import('typeguard._functions', 'check_multi_variable_assignment')
                    targets_arg = List([Dict(keys=list(target), values=list(target.values())) for target in targets], ctx=Load())
                    node.value = Call(func_name, [node.value, targets_arg, self._memo.get_memo_name()], [])
        return node

    def visit_NamedExpr(self, node: NamedExpr) -> Any:
        """This injects a type check into an assignment expression (a := foo())."""
        self.generic_visit(node)
        if isinstance(self._memo.node, (FunctionDef, AsyncFunctionDef)) and isinstance(node.target, Name):
            self._memo.ignored_names.add(node.target.id)
            annotation = self._memo.variable_annotations.get(node.target.id)
            if annotation is None:
                return node
            func_name = self._get_import('typeguard._functions', 'check_variable_assignment')
            node.value = Call(func_name, [node.value, Constant(node.target.id), annotation, self._memo.get_memo_name()], [])
        return node

    def visit_AugAssign(self, node: AugAssign) -> Any:
        """
        This injects a type check into an augmented assignment expression (a += 1).

        """
        self.generic_visit(node)
        if isinstance(self._memo.node, (FunctionDef, AsyncFunctionDef)) and isinstance(node.target, Name):
            annotation = self._memo.variable_annotations.get(node.target.id)
            if annotation is None:
                return node
            try:
                operator_func_name = aug_assign_functions[node.op.__class__]
            except KeyError:
                return node
            operator_func = self._get_import('operator', operator_func_name)
            operator_call = Call(operator_func, [Name(node.target.id, ctx=Load()), node.value], [])
            check_call = Call(self._get_import('typeguard._functions', 'check_variable_assignment'), [operator_call, Constant(node.target.id), annotation, self._memo.get_memo_name()], [])
            return Assign(targets=[node.target], value=check_call)
        return node

    def visit_If(self, node: If) -> Any:
        """
        This blocks names from being collected from a module-level
        "if typing.TYPE_CHECKING:" block, so that they won't be type checked.

        """
        self.generic_visit(node)
        if self._memo is self._module_memo and isinstance(node.test, Name) and self._memo.name_matches(node.test, 'typing.TYPE_CHECKING'):
            collector = NameCollector()
            collector.visit(node)
            self._memo.ignored_names.update(collector.names)
        return node