import importlib.util
import os
import tempfile
from pathlib import PurePath
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Union
import fsspec
import numpy as np
from .utils import logging
from .utils import tqdm as hf_tqdm
def add_documents(self, documents: Union[List[str], 'Dataset'], column: Optional[str]=None):
    """
        Add documents to the index.
        If the documents are inside a certain column, you can specify it using the `column` argument.
        """
    index_name = self.es_index_name
    index_config = self.es_index_config
    self.es_client.indices.create(index=index_name, body=index_config)
    number_of_docs = len(documents)
    progress = hf_tqdm(unit='docs', total=number_of_docs)
    successes = 0

    def passage_generator():
        if column is not None:
            for i, example in enumerate(documents):
                yield {'text': example[column], '_id': i}
        else:
            for i, example in enumerate(documents):
                yield {'text': example, '_id': i}
    import elasticsearch as es
    for ok, action in es.helpers.streaming_bulk(client=self.es_client, index=index_name, actions=passage_generator()):
        progress.update(1)
        successes += ok
    if successes != len(documents):
        logger.warning(f'Some documents failed to be added to ElasticSearch. Failures: {len(documents) - successes}/{len(documents)}')
    logger.info(f'Indexed {successes:d} documents')