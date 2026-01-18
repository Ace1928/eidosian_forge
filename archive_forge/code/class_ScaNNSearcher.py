import os
from typing import Optional, Union
import numpy as np
from huggingface_hub import hf_hub_download
from ... import AutoTokenizer
from ...utils import logging
class ScaNNSearcher:
    """Note that ScaNNSearcher cannot currently be used within the model. In future versions, it might however be included."""

    def __init__(self, db, num_neighbors, dimensions_per_block=2, num_leaves=1000, num_leaves_to_search=100, training_sample_size=100000):
        """Build scann searcher."""
        from scann.scann_ops.py.scann_ops_pybind import builder as Builder
        builder = Builder(db=db, num_neighbors=num_neighbors, distance_measure='dot_product')
        builder = builder.tree(num_leaves=num_leaves, num_leaves_to_search=num_leaves_to_search, training_sample_size=training_sample_size)
        builder = builder.score_ah(dimensions_per_block=dimensions_per_block)
        self.searcher = builder.build()

    def search_batched(self, question_projection):
        retrieved_block_ids, _ = self.searcher.search_batched(question_projection.detach().cpu())
        return retrieved_block_ids.astype('int64')