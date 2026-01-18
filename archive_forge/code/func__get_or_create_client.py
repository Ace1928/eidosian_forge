import logging
from typing import TYPE_CHECKING, Dict, List, Optional
from ray.data.block import Block, BlockMetadata
from ray.data.datasource.datasource import Datasource, ReadTask
from ray.util.annotations import PublicAPI
def _get_or_create_client(self):
    import pymongo
    if self._client is None:
        self._client = pymongo.MongoClient(self._uri)
        _validate_database_collection_exist(self._client, self._database, self._collection)
        self._avg_obj_size = self._client[self._database].command('collstats', self._collection)['avgObjSize']