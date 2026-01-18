from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from mlflow.protos.mlflow_artifacts_pb2 import (
from mlflow.protos.mlflow_artifacts_pb2 import (
@dataclass
class MultipartUploadCredential:
    url: str
    part_number: int
    headers: Dict[str, Any]

    def to_proto(self):
        credential = ProtoMultipartUploadCredential()
        credential.url = self.url
        credential.part_number = self.part_number
        credential.headers.update(self.headers)
        return credential

    @classmethod
    def from_dict(cls, dict_):
        return cls(url=dict_['url'], part_number=dict_['part_number'], headers=dict_.get('headers', {}))