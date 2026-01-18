import ast
import json
import os
import sys
import tokenize
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
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