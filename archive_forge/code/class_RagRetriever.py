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
class RagRetriever:
    """
    Retriever used to get documents from vector queries. It retrieves the documents embeddings as well as the documents
    contents, and it formats them to be used with a RagModel.

    Args:
        config ([`RagConfig`]):
            The configuration of the RAG model this Retriever is used with. Contains parameters indicating which
            `Index` to build. You can load your own custom dataset with `config.index_name="custom"` or use a canonical
            one (default) from the datasets library with `config.index_name="wiki_dpr"` for example.
        question_encoder_tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer that was used to tokenize the question. It is used to decode the question and then use the
            generator_tokenizer.
        generator_tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer used for the generator part of the RagModel.
        index ([`~models.rag.retrieval_rag.Index`], optional, defaults to the one defined by the configuration):
            If specified, use this index instead of the one built using the configuration

    Examples:

    ```python
    >>> # To load the default "wiki_dpr" dataset with 21M passages from wikipedia (index name is 'compressed' or 'exact')
    >>> from transformers import RagRetriever

    >>> retriever = RagRetriever.from_pretrained(
    ...     "facebook/dpr-ctx_encoder-single-nq-base", dataset="wiki_dpr", index_name="compressed"
    ... )

    >>> # To load your own indexed dataset built with the datasets library. More info on how to build the indexed dataset in examples/rag/use_own_knowledge_dataset.py
    >>> from transformers import RagRetriever

    >>> dataset = (
    ...     ...
    ... )  # dataset must be a datasets.Datasets object with columns "title", "text" and "embeddings", and it must have a faiss index
    >>> retriever = RagRetriever.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", indexed_dataset=dataset)

    >>> # To load your own indexed dataset built with the datasets library that was saved on disk. More info in examples/rag/use_own_knowledge_dataset.py
    >>> from transformers import RagRetriever

    >>> dataset_path = "path/to/my/dataset"  # dataset saved via *dataset.save_to_disk(...)*
    >>> index_path = "path/to/my/index.faiss"  # faiss index saved via *dataset.get_index("embeddings").save(...)*
    >>> retriever = RagRetriever.from_pretrained(
    ...     "facebook/dpr-ctx_encoder-single-nq-base",
    ...     index_name="custom",
    ...     passages_path=dataset_path,
    ...     index_path=index_path,
    ... )

    >>> # To load the legacy index built originally for Rag's paper
    >>> from transformers import RagRetriever

    >>> retriever = RagRetriever.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", index_name="legacy")
    ```"""

    def __init__(self, config, question_encoder_tokenizer, generator_tokenizer, index=None, init_retrieval=True):
        self._init_retrieval = init_retrieval
        requires_backends(self, ['datasets', 'faiss'])
        super().__init__()
        self.index = index or self._build_index(config)
        self.generator_tokenizer = generator_tokenizer
        self.question_encoder_tokenizer = question_encoder_tokenizer
        self.n_docs = config.n_docs
        self.batch_size = config.retrieval_batch_size
        self.config = config
        if self._init_retrieval:
            self.init_retrieval()
        self.ctx_encoder_tokenizer = None
        self.return_tokenized_docs = False

    @staticmethod
    def _build_index(config):
        if config.index_name == 'legacy':
            return LegacyIndex(config.retrieval_vector_size, config.index_path or LEGACY_INDEX_PATH)
        elif config.index_name == 'custom':
            return CustomHFIndex.load_from_disk(vector_size=config.retrieval_vector_size, dataset_path=config.passages_path, index_path=config.index_path)
        else:
            return CanonicalHFIndex(vector_size=config.retrieval_vector_size, dataset_name=config.dataset, dataset_split=config.dataset_split, index_name=config.index_name, index_path=config.index_path, use_dummy_dataset=config.use_dummy_dataset)

    @classmethod
    def from_pretrained(cls, retriever_name_or_path, indexed_dataset=None, **kwargs):
        requires_backends(cls, ['datasets', 'faiss'])
        config = kwargs.pop('config', None) or RagConfig.from_pretrained(retriever_name_or_path, **kwargs)
        rag_tokenizer = RagTokenizer.from_pretrained(retriever_name_or_path, config=config)
        question_encoder_tokenizer = rag_tokenizer.question_encoder
        generator_tokenizer = rag_tokenizer.generator
        if indexed_dataset is not None:
            config.index_name = 'custom'
            index = CustomHFIndex(config.retrieval_vector_size, indexed_dataset)
        else:
            index = cls._build_index(config)
        return cls(config, question_encoder_tokenizer=question_encoder_tokenizer, generator_tokenizer=generator_tokenizer, index=index)

    def save_pretrained(self, save_directory):
        if isinstance(self.index, CustomHFIndex):
            if self.config.index_path is None:
                index_path = os.path.join(save_directory, 'hf_dataset_index.faiss')
                self.index.dataset.get_index('embeddings').save(index_path)
                self.config.index_path = index_path
            if self.config.passages_path is None:
                passages_path = os.path.join(save_directory, 'hf_dataset')
                faiss_index = self.index.dataset._indexes.pop('embeddings')
                self.index.dataset.save_to_disk(passages_path)
                self.index.dataset._indexes['embeddings'] = faiss_index
                self.config.passages_path = passages_path
        self.config.save_pretrained(save_directory)
        rag_tokenizer = RagTokenizer(question_encoder=self.question_encoder_tokenizer, generator=self.generator_tokenizer)
        rag_tokenizer.save_pretrained(save_directory)

    def init_retrieval(self):
        """
        Retriever initialization function. It loads the index into memory.
        """
        logger.info('initializing retrieval')
        self.index.init_index()

    def postprocess_docs(self, docs, input_strings, prefix, n_docs, return_tensors=None):
        """
        Postprocessing retrieved `docs` and combining them with `input_strings`.

        Args:
            docs  (`dict`):
                Retrieved documents.
            input_strings (`str`):
                Input strings decoded by `preprocess_query`.
            prefix (`str`):
                Prefix added at the beginning of each input, typically used with T5-based models.

        Return:
            `tuple(tensors)`: a tuple consisting of two elements: contextualized `input_ids` and a compatible
            `attention_mask`.
        """

        def cat_input_and_doc(doc_title, doc_text, input_string, prefix):
            if doc_title.startswith('"'):
                doc_title = doc_title[1:]
            if doc_title.endswith('"'):
                doc_title = doc_title[:-1]
            if prefix is None:
                prefix = ''
            out = (prefix + doc_title + self.config.title_sep + doc_text + self.config.doc_sep + input_string).replace('  ', ' ')
            return out
        rag_input_strings = [cat_input_and_doc(docs[i]['title'][j], docs[i]['text'][j], input_strings[i], prefix) for i in range(len(docs)) for j in range(n_docs)]
        contextualized_inputs = self.generator_tokenizer.batch_encode_plus(rag_input_strings, max_length=self.config.max_combined_length, return_tensors=return_tensors, padding='max_length', truncation=True)
        return (contextualized_inputs['input_ids'], contextualized_inputs['attention_mask'])

    def _chunk_tensor(self, t: Iterable, chunk_size: int) -> List[Iterable]:
        return [t[i:i + chunk_size] for i in range(0, len(t), chunk_size)]

    def _main_retrieve(self, question_hidden_states: np.ndarray, n_docs: int) -> Tuple[np.ndarray, np.ndarray]:
        question_hidden_states_batched = self._chunk_tensor(question_hidden_states, self.batch_size)
        ids_batched = []
        vectors_batched = []
        for question_hidden_states in question_hidden_states_batched:
            start_time = time.time()
            ids, vectors = self.index.get_top_docs(question_hidden_states, n_docs)
            logger.debug(f'index search time: {time.time() - start_time} sec, batch size {question_hidden_states.shape}')
            ids_batched.extend(ids)
            vectors_batched.extend(vectors)
        return (np.array(ids_batched), np.array(vectors_batched))

    def retrieve(self, question_hidden_states: np.ndarray, n_docs: int) -> Tuple[np.ndarray, List[dict]]:
        """
        Retrieves documents for specified `question_hidden_states`.

        Args:
            question_hidden_states (`np.ndarray` of shape `(batch_size, vector_size)`):
                A batch of query vectors to retrieve with.
            n_docs (`int`):
                The number of docs retrieved per query.

        Return:
            `Tuple[np.ndarray, np.ndarray, List[dict]]`: A tuple with the following objects:

            - **retrieved_doc_embeds** (`np.ndarray` of shape `(batch_size, n_docs, dim)`) -- The retrieval embeddings
              of the retrieved docs per query.
            - **doc_ids** (`np.ndarray` of shape `(batch_size, n_docs)`) -- The ids of the documents in the index
            - **doc_dicts** (`List[dict]`): The `retrieved_doc_embeds` examples per query.
        """
        doc_ids, retrieved_doc_embeds = self._main_retrieve(question_hidden_states, n_docs)
        return (retrieved_doc_embeds, doc_ids, self.index.get_doc_dicts(doc_ids))

    def set_ctx_encoder_tokenizer(self, ctx_encoder_tokenizer: PreTrainedTokenizer):
        self.ctx_encoder_tokenizer = ctx_encoder_tokenizer
        self.return_tokenized_docs = True

    def __call__(self, question_input_ids: List[List[int]], question_hidden_states: np.ndarray, prefix=None, n_docs=None, return_tensors=None) -> BatchEncoding:
        """
        Retrieves documents for specified `question_hidden_states`.

        Args:
            question_input_ids (`List[List[int]]`) batch of input ids
            question_hidden_states (`np.ndarray` of shape `(batch_size, vector_size)`:
                A batch of query vectors to retrieve with.
            prefix (`str`, *optional*):
                The prefix used by the generator's tokenizer.
            n_docs (`int`, *optional*):
                The number of docs retrieved per query.
            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to "pt"):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.

        Returns: [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:

            - **context_input_ids** -- List of token ids to be fed to a model.

              [What are input IDs?](../glossary#input-ids)

            - **context_attention_mask** -- List of indices specifying which tokens should be attended to by the model
            (when `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names`).

              [What are attention masks?](../glossary#attention-mask)

            - **retrieved_doc_embeds** -- List of embeddings of the retrieved documents
            - **doc_ids** -- List of ids of the retrieved documents
        """
        n_docs = n_docs if n_docs is not None else self.n_docs
        prefix = prefix if prefix is not None else self.config.generator.prefix
        retrieved_doc_embeds, doc_ids, docs = self.retrieve(question_hidden_states, n_docs)
        input_strings = self.question_encoder_tokenizer.batch_decode(question_input_ids, skip_special_tokens=True)
        context_input_ids, context_attention_mask = self.postprocess_docs(docs, input_strings, prefix, n_docs, return_tensors=return_tensors)
        if self.return_tokenized_docs:
            retrieved_doc_text = []
            retrieved_doc_title = []
            for b_idx in range(len(docs)):
                for doc_idx in range(n_docs):
                    retrieved_doc_text.append(docs[b_idx]['text'][doc_idx])
                    retrieved_doc_title.append(docs[b_idx]['title'][doc_idx])
            tokenized_docs = self.ctx_encoder_tokenizer(retrieved_doc_title, retrieved_doc_text, truncation=True, padding='longest', return_tensors=return_tensors)
            return BatchEncoding({'context_input_ids': context_input_ids, 'context_attention_mask': context_attention_mask, 'retrieved_doc_embeds': retrieved_doc_embeds, 'doc_ids': doc_ids, 'tokenized_doc_ids': tokenized_docs['input_ids'], 'tokenized_doc_attention_mask': tokenized_docs['attention_mask']}, tensor_type=return_tensors)
        else:
            return BatchEncoding({'context_input_ids': context_input_ids, 'context_attention_mask': context_attention_mask, 'retrieved_doc_embeds': retrieved_doc_embeds, 'doc_ids': doc_ids}, tensor_type=return_tensors)