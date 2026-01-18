from __future__ import annotations
import dataclasses
from typing import List, Literal, Optional
from torch.onnx._internal.diagnostics.infra.sarif import (
@dataclasses.dataclass
class SarifLog(object):
    """Static Analysis Results Format (SARIF) Version 2.1.0 JSON Schema: a standard format for the output of static analysis tools."""
    runs: List[_run.Run] = dataclasses.field(metadata={'schema_property_name': 'runs'})
    version: Literal['2.1.0'] = dataclasses.field(metadata={'schema_property_name': 'version'})
    schema_uri: Optional[str] = dataclasses.field(default=None, metadata={'schema_property_name': '$schema'})
    inline_external_properties: Optional[List[_external_properties.ExternalProperties]] = dataclasses.field(default=None, metadata={'schema_property_name': 'inlineExternalProperties'})
    properties: Optional[_property_bag.PropertyBag] = dataclasses.field(default=None, metadata={'schema_property_name': 'properties'})