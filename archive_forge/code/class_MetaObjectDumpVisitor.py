import ast
import json
import os
import sys
import tokenize
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
class MetaObjectDumpVisitor(ast.NodeVisitor):
    """AST visitor for parsing sources and creating the data structure for
       JSON."""

    def __init__(self, context: VisitorContext):
        super().__init__()
        self._context = context
        self._json_class_list: ClassList = []
        self._properties: List[PropertyEntry] = []
        self._signals: List[Signal] = []
        self._within_class: bool = False
        self._qt_modules: Set[str] = set()
        self._qml_import_name = ''
        self._qml_import_major_version = 0
        self._qml_import_minor_version = 0

    def json_class_list(self) -> ClassList:
        return self._json_class_list

    def qml_import_name(self) -> str:
        return self._qml_import_name

    def qml_import_version(self) -> Tuple[int, int]:
        return (self._qml_import_major_version, self._qml_import_minor_version)

    def qt_modules(self):
        return sorted(self._qt_modules)

    @staticmethod
    def create_ast(filename: Path) -> ast.Module:
        """Create an Abstract Syntax Tree on which a visitor can be run"""
        node = None
        with tokenize.open(filename) as file:
            node = ast.parse(file.read(), mode='exec')
        return node

    def visit_Assign(self, node: ast.Assign):
        """Parse the global constants for QML-relevant values"""
        var_name, value_node = _parse_assignment(node)
        if not var_name or not isinstance(value_node, ast.Constant):
            return
        value = value_node.value
        if var_name == QML_IMPORT_NAME:
            self._qml_import_name = value
        elif var_name == QML_IMPORT_MAJOR_VERSION:
            self._qml_import_major_version = value
        elif var_name == QML_IMPORT_MINOR_VERSION:
            self._qml_import_minor_version = value

    def visit_ClassDef(self, node: ast.Module):
        """Visit a class definition"""
        self._properties = []
        self._signals = []
        self._slots = []
        self._within_class = True
        qualified_name = node.name
        last_dot = qualified_name.rfind('.')
        name = qualified_name[last_dot + 1:] if last_dot != -1 else qualified_name
        data = {'className': name, 'qualifiedClassName': qualified_name}
        q_object = False
        bases = []
        for b in node.bases:
            if isinstance(b, ast.Name):
                base_name = _name(b)
                if base_name in self._context.qobject_derived:
                    q_object = True
                    self._context.qobject_derived.append(name)
                base_dict = {'access': 'public', 'name': base_name}
                bases.append(base_dict)
        data['object'] = q_object
        if bases:
            data['superClasses'] = bases
        class_decorators: List[dict] = []
        for d in node.decorator_list:
            self._parse_class_decorator(d, class_decorators)
        if class_decorators:
            data['classInfos'] = class_decorators
        for b in node.body:
            if isinstance(b, ast.Assign):
                self._parse_class_variable(b)
            else:
                self.visit(b)
        if self._properties:
            data['properties'] = self._properties
        if self._signals:
            data['signals'] = self._signals
        if self._slots:
            data['slots'] = self._slots
        self._json_class_list.append(data)
        self._within_class = False

    def visit_FunctionDef(self, node):
        if self._within_class:
            for d in node.decorator_list:
                self._parse_function_decorator(node.name, d)

    def _parse_class_decorator(self, node: AstDecorator, class_decorators: List[dict]):
        """Parse ClassInfo decorators."""
        if isinstance(node, ast.Call):
            name = _func_name(node)
            if name == 'QmlUncreatable':
                class_decorators.append(_decorator('QML.Creatable', 'false'))
                if node.args:
                    reason = node.args[0].value
                    if isinstance(reason, str):
                        d = _decorator('QML.UncreatableReason', reason)
                        class_decorators.append(d)
            elif name == 'QmlAttached' and len(node.args) == 1:
                d = _decorator('QML.Attached', node.args[0].id)
                class_decorators.append(d)
            elif name == 'QmlExtended' and len(node.args) == 1:
                d = _decorator('QML.Extended', node.args[0].id)
                class_decorators.append(d)
            elif name == 'ClassInfo' and node.keywords:
                kw = node.keywords[0]
                class_decorators.append(_decorator(kw.arg, kw.value.value))
            elif name == 'QmlForeign' and len(node.args) == 1:
                d = _decorator('QML.Foreign', node.args[0].id)
                class_decorators.append(d)
            elif name == 'QmlNamedElement' and node.args:
                name = node.args[0].value
                class_decorators.append(_decorator('QML.Element', name))
            else:
                print('Unknown decorator with parameters:', name, file=sys.stderr)
            return
        if isinstance(node, ast.Name):
            name = node.id
            if name == 'QmlElement':
                class_decorators.append(_decorator('QML.Element', 'auto'))
            elif name == 'QmlSingleton':
                class_decorators.append(_decorator('QML.Singleton', 'true'))
            elif name == 'QmlAnonymous':
                class_decorators.append(_decorator('QML.Element', 'anonymous'))
            else:
                print('Unknown decorator:', name, file=sys.stderr)
            return

    def _index_of_property(self, name: str) -> int:
        """Search a property by name"""
        for i in range(len(self._properties)):
            if self._properties[i]['name'] == name:
                return i
        return -1

    def _create_property_entry(self, name: str, type: str, getter: Optional[str]=None) -> PropertyEntry:
        """Create a property JSON entry."""
        result: PropertyEntry = {'name': name, 'type': type, 'index': len(self._properties)}
        if getter:
            result['read'] = getter
        return result

    def _parse_function_decorator(self, func_name: str, node: AstDecorator):
        """Parse function decorators."""
        if isinstance(node, ast.Attribute):
            name = node.value.id
            value = node.attr
            if value == 'setter':
                idx = self._index_of_property(name)
                if idx != -1:
                    self._properties[idx]['write'] = func_name
            return
        if isinstance(node, ast.Call):
            name = _name(node.func)
            if name == 'Property':
                if node.args:
                    type = _python_to_cpp_type(_name(node.args[0]))
                    prop = self._create_property_entry(func_name, type, func_name)
                    _parse_property_kwargs(node.keywords, prop)
                    self._properties.append(prop)
            elif name == 'Slot':
                self._slots.append(_parse_slot(func_name, node))
            else:
                print('Unknown decorator with parameters:', name, file=sys.stderr)

    def _parse_class_variable(self, node: ast.Assign):
        """Parse a class variable assignment (Property, Signal, etc.)"""
        var_name, call = _parse_assignment(node)
        if not var_name or not isinstance(node.value, ast.Call):
            return
        func_name = _func_name(call)
        if func_name == 'Signal' or func_name == 'QtCore.Signal':
            signal: Signal = {'access': 'public', 'name': var_name, 'arguments': _parse_call_args(call), 'returnType': 'void'}
            self._signals.append(signal)
        elif func_name == 'Property' or func_name == 'QtCore.Property':
            type = _python_to_cpp_type(call.args[0].id)
            prop = self._create_property_entry(var_name, type, call.args[1].id)
            if len(call.args) > 2:
                prop['write'] = call.args[2].id
            _parse_property_kwargs(call.keywords, prop)
            self._properties.append(prop)
        elif func_name == 'ListProperty' or func_name == 'QtCore.ListProperty':
            type = _python_to_cpp_type(call.args[0].id)
            type = f'QQmlListProperty<{type}>'
            prop = self._create_property_entry(var_name, type)
            self._properties.append(prop)

    def visit_Import(self, node):
        for n in node.names:
            self._handle_import(n.name)

    def visit_ImportFrom(self, node):
        if '.' in node.module:
            self._handle_import(node.module)
        elif node.module == 'PySide6':
            for n in node.names:
                if n.name.startswith('Qt'):
                    self._qt_modules.add(n.name)

    def _handle_import(self, mod: str):
        if mod.startswith('PySide6.'):
            self._qt_modules.add(mod[8:])