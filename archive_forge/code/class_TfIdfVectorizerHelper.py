from typing import Any
import numpy as np
import onnx
from onnx import NodeProto
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
class TfIdfVectorizerHelper:

    def __init__(self, **params: Any) -> None:
        mode = 'mode'
        min_gram_length = 'min_gram_length'
        max_gram_length = 'max_gram_length'
        max_skip_count = 'max_skip_count'
        ngram_counts = 'ngram_counts'
        ngram_indexes = 'ngram_indexes'
        pool_int64s = 'pool_int64s'
        required_attr = [mode, min_gram_length, max_gram_length, max_skip_count, ngram_counts, ngram_indexes, pool_int64s]
        for i in required_attr:
            assert i in params, f'Missing attribute: {i}'
        self.mode = params[mode]
        self.min_gram_length = params[min_gram_length]
        self.max_gram_length = params[max_gram_length]
        self.max_skip_count = params[max_skip_count]
        self.ngram_counts = params[ngram_counts]
        self.ngram_indexes = params[ngram_indexes]
        self.pool_int64s = params[pool_int64s]

    def make_node_noweights(self) -> NodeProto:
        return onnx.helper.make_node('TfIdfVectorizer', inputs=['X'], outputs=['Y'], mode=self.mode, min_gram_length=self.min_gram_length, max_gram_length=self.max_gram_length, max_skip_count=self.max_skip_count, ngram_counts=self.ngram_counts, ngram_indexes=self.ngram_indexes, pool_int64s=self.pool_int64s)