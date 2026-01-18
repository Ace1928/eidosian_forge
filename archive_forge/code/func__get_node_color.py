from __future__ import annotations
import os
import inspect
import tempfile
from qiskit.utils import optionals as _optionals
from qiskit.passmanager.base_tasks import BaseController, GenericPass
from qiskit.passmanager.flow_controllers import FlowControllerLinear
from qiskit.transpiler.basepasses import AnalysisPass, TransformationPass
from .exceptions import VisualizationError
def _get_node_color(pss, style):
    for typ, color in style.items():
        if isinstance(pss, typ):
            return color
    for typ, color in DEFAULT_STYLE.items():
        if isinstance(pss, typ):
            return color
    return 'black'