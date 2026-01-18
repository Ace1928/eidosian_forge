from __future__ import annotations
import dataclasses
from typing import Optional
from torch.onnx._internal.diagnostics.infra.sarif import (
@dataclasses.dataclass
class ArtifactContent(object):
    """Represents the contents of an artifact."""
    binary: Optional[str] = dataclasses.field(default=None, metadata={'schema_property_name': 'binary'})
    properties: Optional[_property_bag.PropertyBag] = dataclasses.field(default=None, metadata={'schema_property_name': 'properties'})
    rendered: Optional[_multiformat_message_string.MultiformatMessageString] = dataclasses.field(default=None, metadata={'schema_property_name': 'rendered'})
    text: Optional[str] = dataclasses.field(default=None, metadata={'schema_property_name': 'text'})