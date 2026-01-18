import os
import pickle
import time
from typing import Iterable, List, Optional, Tuple
import numpy as np
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import BatchEncoding
from ...utils import cached_file, is_datasets_available, is_faiss_available, logging, requires_backends, strtobool
from .configuration_rag import RagConfig
from .tokenization_rag import RagTokenizer
class LegacyIndex(Index):
    """
    An index which can be deserialized from the files built using https://github.com/facebookresearch/DPR. We use
    default faiss index parameters as specified in that repository.

    Args:
        vector_size (`int`):
            The dimension of indexed vectors.
        index_path (`str`):
            A path to a *directory* containing index files compatible with [`~models.rag.retrieval_rag.LegacyIndex`]
    """
    INDEX_FILENAME = 'hf_bert_base.hnswSQ8_correct_phi_128.c_index'
    PASSAGE_FILENAME = 'psgs_w100.tsv.pkl'

    def __init__(self, vector_size, index_path):
        self.index_id_to_db_id = []
        self.index_path = index_path
        self.passages = self._load_passages()
        self.vector_size = vector_size
        self.index = None
        self._index_initialized = False

    def _resolve_path(self, index_path, filename):
        is_local = os.path.isdir(index_path)
        try:
            resolved_archive_file = cached_file(index_path, filename)
        except EnvironmentError:
            msg = f"Can't load '{filename}'. Make sure that:\n\n- '{index_path}' is a correct remote path to a directory containing a file named {filename}\n\n- or '{index_path}' is the correct path to a directory containing a file named {filename}.\n\n"
            raise EnvironmentError(msg)
        if is_local:
            logger.info(f'loading file {resolved_archive_file}')
        else:
            logger.info(f'loading file {filename} from cache at {resolved_archive_file}')
        return resolved_archive_file

    def _load_passages(self):
        logger.info(f'Loading passages from {self.index_path}')
        passages_path = self._resolve_path(self.index_path, self.PASSAGE_FILENAME)
        if not strtobool(os.environ.get('TRUST_REMOTE_CODE', 'False')):
            raise ValueError("This part uses `pickle.load` which is insecure and will execute arbitrary code that is potentially malicious. It's recommended to never unpickle data that could have come from an untrusted source, or that could have been tampered with. If you already verified the pickle data and decided to use it, you can set the environment variable `TRUST_REMOTE_CODE` to `True` to allow it.")
        with open(passages_path, 'rb') as passages_file:
            passages = pickle.load(passages_file)
        return passages

    def _deserialize_index(self):
        logger.info(f'Loading index from {self.index_path}')
        resolved_index_path = self._resolve_path(self.index_path, self.INDEX_FILENAME + '.index.dpr')
        self.index = faiss.read_index(resolved_index_path)
        resolved_meta_path = self._resolve_path(self.index_path, self.INDEX_FILENAME + '.index_meta.dpr')
        if not strtobool(os.environ.get('TRUST_REMOTE_CODE', 'False')):
            raise ValueError("This part uses `pickle.load` which is insecure and will execute arbitrary code that is potentially malicious. It's recommended to never unpickle data that could have come from an untrusted source, or that could have been tampered with. If you already verified the pickle data and decided to use it, you can set the environment variable `TRUST_REMOTE_CODE` to `True` to allow it.")
        with open(resolved_meta_path, 'rb') as metadata_file:
            self.index_id_to_db_id = pickle.load(metadata_file)
        assert len(self.index_id_to_db_id) == self.index.ntotal, 'Deserialized index_id_to_db_id should match faiss index size'

    def is_initialized(self):
        return self._index_initialized

    def init_index(self):
        index = faiss.IndexHNSWFlat(self.vector_size + 1, 512)
        index.hnsw.efSearch = 128
        index.hnsw.efConstruction = 200
        self.index = index
        self._deserialize_index()
        self._index_initialized = True

    def get_doc_dicts(self, doc_ids: np.array):
        doc_list = []
        for doc_ids_i in doc_ids:
            ids = [str(int(doc_id)) for doc_id in doc_ids_i]
            docs = [self.passages[doc_id] for doc_id in ids]
            doc_list.append(docs)
        doc_dicts = []
        for docs in doc_list:
            doc_dict = {}
            doc_dict['title'] = [doc[1] for doc in docs]
            doc_dict['text'] = [doc[0] for doc in docs]
            doc_dicts.append(doc_dict)
        return doc_dicts

    def get_top_docs(self, question_hidden_states: np.ndarray, n_docs=5) -> Tuple[np.ndarray, np.ndarray]:
        aux_dim = np.zeros(len(question_hidden_states), dtype='float32').reshape(-1, 1)
        query_nhsw_vectors = np.hstack((question_hidden_states, aux_dim))
        _, docs_ids = self.index.search(query_nhsw_vectors, n_docs)
        vectors = [[self.index.reconstruct(int(doc_id))[:-1] for doc_id in doc_ids] for doc_ids in docs_ids]
        ids = [[int(self.index_id_to_db_id[doc_id]) for doc_id in doc_ids] for doc_ids in docs_ids]
        return (np.array(ids), np.array(vectors))