from __future__ import annotations
import dataclasses
from typing import List, Optional
from torch.onnx._internal.diagnostics.infra.sarif import (
@dataclasses.dataclass
class ArtifactChange(object):
    """A change to a single artifact."""
    artifact_location: _artifact_location.ArtifactLocation = dataclasses.field(metadata={'schema_property_name': 'artifactLocation'})
    replacements: List[_replacement.Replacement] = dataclasses.field(metadata={'schema_property_name': 'replacements'})
    properties: Optional[_property_bag.PropertyBag] = dataclasses.field(default=None, metadata={'schema_property_name': 'properties'})