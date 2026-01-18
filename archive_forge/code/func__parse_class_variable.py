import ast
import json
import os
import sys
import tokenize
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
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