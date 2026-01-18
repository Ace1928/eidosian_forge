import csv
import datetime
import os
from collections import OrderedDict, defaultdict
from typing import IO, Any, Dict, List, Optional, Set
from tabulate import tabulate
import onnx
from onnx import GraphProto, defs, helper
class NodeCoverage:

    def __init__(self) -> None:
        self.op_type: Optional[str] = None
        self.attr_coverages: Dict[str, AttrCoverage] = defaultdict(AttrCoverage)

    def add(self, node: onnx.NodeProto) -> None:
        assert self.op_type in [None, node.op_type]
        if self.op_type is None:
            self.op_type = node.op_type
            assert self.op_type is not None
            self.schema = defs.get_schema(self.op_type, domain=node.domain)
        for attr in node.attribute:
            self.attr_coverages[attr.name].add(attr)