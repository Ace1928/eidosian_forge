from __future__ import annotations
import dataclasses
from typing import List, Optional
from torch.onnx._internal.diagnostics.infra.sarif import (
@dataclasses.dataclass
class ReportingDescriptorRelationship(object):
    """Information about the relation of one reporting descriptor to another."""
    target: _reporting_descriptor_reference.ReportingDescriptorReference = dataclasses.field(metadata={'schema_property_name': 'target'})
    description: Optional[_message.Message] = dataclasses.field(default=None, metadata={'schema_property_name': 'description'})
    kinds: List[str] = dataclasses.field(default_factory=lambda: ['relevant'], metadata={'schema_property_name': 'kinds'})
    properties: Optional[_property_bag.PropertyBag] = dataclasses.field(default=None, metadata={'schema_property_name': 'properties'})