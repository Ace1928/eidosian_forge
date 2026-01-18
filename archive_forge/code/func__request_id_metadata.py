from typing import Any, Dict, List, Optional, Tuple
import grpc
from ray.util.annotations import DeveloperAPI, PublicAPI
@DeveloperAPI
def _request_id_metadata(self) -> List[Tuple[str, str]]:
    for key, value in self._trailing_metadata:
        if key == 'request_id':
            return [(key, value)]
    return []