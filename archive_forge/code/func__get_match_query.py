import logging
from typing import TYPE_CHECKING, Dict, List, Optional
from ray.data.block import Block, BlockMetadata
from ray.data.datasource.datasource import Datasource, ReadTask
from ray.util.annotations import PublicAPI
def _get_match_query(self, pipeline: List[Dict]) -> Dict:
    if len(pipeline) == 0 or '$match' not in pipeline[0]:
        return {}
    return pipeline[0]['$match']