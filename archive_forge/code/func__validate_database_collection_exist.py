import logging
from typing import TYPE_CHECKING, Dict, List, Optional
from ray.data.block import Block, BlockMetadata
from ray.data.datasource.datasource import Datasource, ReadTask
from ray.util.annotations import PublicAPI
def _validate_database_collection_exist(client, database: str, collection: str):
    db_names = client.list_database_names()
    if database not in db_names:
        raise ValueError(f"The destination database {database} doesn't exist.")
    collection_names = client[database].list_collection_names()
    if collection not in collection_names:
        raise ValueError(f"The destination collection {collection} doesn't exist.")