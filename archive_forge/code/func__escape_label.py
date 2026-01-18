import argparse
import json
from collections import defaultdict
from typing import Any, Callable, Dict, Optional
import pydot
from onnx import GraphProto, ModelProto, NodeProto
def _escape_label(name: str) -> str:
    return json.dumps(name)