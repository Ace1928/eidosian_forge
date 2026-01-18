from __future__ import annotations
import os
import sys
from typing import Any, Iterable
import numpy as np
import onnx
import onnx.external_data_helper as ext_data
import onnx.helper
import onnx.onnx_cpp2py_export.checker as c_checker
def _enumerate_subgraphs(graph):
    for node in graph.node:
        for att in node.attribute:
            if att.g:
                yield att.g
                yield from _enumerate_subgraphs(att.g)