from __future__ import annotations

import os
import pickle
import logging
from collections import Counter, defaultdict, OrderedDict
from typing import Iterable, List, Optional, Dict, Any, DefaultDict, Tuple

import nltk  # type: ignore
from nltk.corpus import wordnet as wn  # type: ignore

import spacy
from spacy.language import Language
from spacy.tokens import Doc
from typing import cast

from base_tokenizer import CharTokenizer

import concurrent.futures
from tqdm import tqdm

import sentencepiece as spm  # type: ignore

# Optional: For embedding models; use gensim if available.
try:
    from gensim.models import KeyedVectors  # type: ignore
except ImportError:
    KeyedVectors = None

from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments  # type: ignore
import torch
from torch.utils.data import TensorDataset

from pathlib import Path
import threading
import time


# ---------------------------------------------------------------------------
# Utility: Simple LRU Cache implementation with a fixed max size.
class LRUCache(dict):
    def __init__(self, maxsize: int = 4096, *args, **kwargs):
        self.maxsize = maxsize
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        elif len(self) >= self.maxsize:
            oldest = next(iter(self))
            del self[oldest]
        super().__setitem__(key, value)

    def clear(self):
        super().clear()


# ---------------------------------------------------------------------------
# Trie data structures for efficient prefix‐based token searches.
class TrieNode:
    """
    A node in the Trie structure used for efficient prefix‐based token searches.
    """

    def __init__(self) -> None:
        self.children: Dict[str, TrieNode] = {}
        self.is_end: bool = False
        self.token: Optional[str] = None  # Holds the complete token at terminal nodes.


class Trie:
    """
    Trie data structure for fast, greedy matching of token sequences.
    """

    def __init__(self) -> None:
        self.root: TrieNode = TrieNode()
        self.search_cache: LRUCache = LRUCache(maxsize=4096)

    def insert(self, token: str) -> None:
        """
        Insert a token into the Trie.
        """
        node: TrieNode = self.root
        for char in token:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.token = token
        # Clear the search cache as the trie structure has changed.
        self.search_cache.clear()
        logger.debug(f"Inserted token into trie: {token}")

    def search_longest(self, text: str, start_index: int) -> str:
        """
        Search for the longest token in the Trie starting from 'start_index' in the given text.
        """
        cache_key = (text, start_index)
        if cache_key in self.search_cache:
            cached_result = self.search_cache[cache_key]
            logger.debug(f"Cached trie result for {cache_key}: {cached_result}")
            return cached_result

        node: TrieNode = self.root
        longest: Optional[str] = None
        i: int = start_index
        while i < len(text) and text[i] in node.children:
            node = node.children[text[i]]
            if node.is_end:
                longest = node.token
            i += 1
        result = longest if longest is not None else ""
        self.search_cache[cache_key] = result
        if longest:
            logger.debug(f"Trie matched '{longest}' at index {start_index}.")
        else:
            logger.debug(f"No match found in trie starting at index {start_index}.")
        return result


# ---------------------------------------------------------------------------
# Modular Component: Text Processing (Normalization and spaCy tokenization)
class TextProcessor:
    def __init__(
        self,
        mode: str,
        lowercase: bool,
        nlp: Optional[Language],
        tokenize_cache: LRUCache,
    ):
        self.mode = mode
        self.lowercase = lowercase
        self.nlp = nlp
        self.tokenize_cache = tokenize_cache

    def normalize(self, text: str) -> str:
        """Helper method to normalize text based on the lowercase flag."""
        return text.lower() if self.lowercase else text

    def get_doc(self, text: str) -> Optional[Doc]:
        """Tokenize text using spaCy with caching."""
        if text in self.tokenize_cache:
            return self.tokenize_cache[text]
        try:
            doc: Optional[Doc] = self.nlp(text) if self.nlp else None
        except Exception as e:
            logger.warning("Tokenization error for text %r: %s", text, e)
            doc = None
        if doc is not None:
            self.tokenize_cache[text] = doc
        return doc


# ---------------------------------------------------------------------------
# Modular Component: N-Gram Extraction
class NGramExtractor:
    def __init__(self, max_ngram: int, cache: LRUCache):
        self.max_ngram = max_ngram
        self.cache = cache

    def extract(self, tokens: List[str], joiner: str) -> Counter[str]:
        """
        Helper method to extract n-grams from a list of tokens.
        """
        cache_key = (joiner, tuple(tokens))
        if cache_key in self.cache:
            return self.cache[cache_key]
        counts: Counter[str] = Counter()
        token_count: int = len(tokens)
        for ngram_size in range(2, self.max_ngram + 1):
            if token_count >= ngram_size:
                for i in range(token_count - ngram_size + 1):
                    ngram: str = joiner.join(tokens[i : i + ngram_size])
                    counts[ngram] += 1
        self.cache[cache_key] = counts
        return counts

    def process_text(self, text: str, text_processor: TextProcessor) -> Counter[str]:
        """
        Process a text sample to extract n-gram counts.
        """
        local_counts: Counter[str] = Counter()
        proc_text: str = text_processor.normalize(text)
        if text_processor.mode == "word" and text_processor.nlp:
            doc: Optional[Doc] = text_processor.get_doc(proc_text)
            if doc is not None:
                word_tokens = [token.text for token in doc if not token.is_space]
            else:
                logger.warning(
                    "spaCy tokenization error in parallel processing; falling back to simple split."
                )
                word_tokens = proc_text.split()
            char_tokens = list(proc_text)
            local_counts.update(self.extract(word_tokens, " "))
            local_counts.update(self.extract(char_tokens, ""))
        else:
            tokens = list(proc_text)
            local_counts.update(self.extract(tokens, ""))
        return local_counts


# ---------------------------------------------------------------------------
# Modular Component: Metadata Population
class MetadataPopulator:
    def __init__(
        self,
        nlp: Optional[Language],
        transformer_model: Optional[Any],
        transformer_tokenizer: Optional[Any],
        embedding_cache: LRUCache,
    ):
        self.nlp = nlp
        self.transformer_model = transformer_model
        self.transformer_tokenizer = transformer_tokenizer
        self.embedding_cache = embedding_cache

    def populate(
        self,
        token: str,
        existing_metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Populate optional metadata for a token from various linguistic resources.
        """
        metadata = existing_metadata if existing_metadata is not None else {}
        # Get WordNet definitions.
        if token.isalpha():
            synsets = wn.synsets(token)
            definitions = [syn.definition() for syn in synsets if syn]
            if definitions:
                metadata.setdefault("wordnet_definitions", definitions)
        # Wiktionary integration using wiktionaryparser.
        try:
            from wiktionaryparser import WiktionaryParser  # type: ignore

            parser = WiktionaryParser()
            wiktionary_data = parser.fetch(token)
            if wiktionary_data:
                definitions = []
                for entry in wiktionary_data:
                    for definition in entry.get("definitions", []):
                        if "text" in definition:
                            definitions.extend(definition["text"])
                if definitions:
                    metadata["wiktionary"] = definitions
        except Exception as e:
            logger.error("Error fetching Wiktionary data for token %s: %s", token, e)
        # Morphological analysis using spaCy.
        if self.nlp:
            doc: Optional[Doc] = self.nlp(token)
            if doc:
                for tok in doc:
                    metadata["morphology"] = tok.morph.to_dict()
        # Populate transformer-based embeddings.
        if (
            self.transformer_model is not None
            and self.transformer_tokenizer is not None
        ):
            if token not in self.embedding_cache:
                inputs = self.transformer_tokenizer(token, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.transformer_model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
                self.embedding_cache[token] = embedding
            metadata["embedding"] = self.embedding_cache[token]
        else:
            logger.warning(
                "Transformer objects are not initialized; skipping embedding population for token: %s",
                token,
            )
        logger.debug(f"Populated metadata for token: {token}")
        return metadata


# ---------------------------------------------------------------------------
# Modular Component: Subword Merging (BPE, Contextual BPE, SentencePiece, Unigram LM)
class SubwordMerger:
    def __init__(self, vocab_builder: VocabularyBuilder):
        self.builder = vocab_builder

    def apply_bpe(self, num_merges: int = 500) -> None:
        """
        Apply a Byte-Pair Encoding (BPE) inspired subword merging algorithm.
        """
        logger.info("Starting BPE subword merging process...")
        bpe_vocab: Dict[str, int] = {
            token: freq
            for token, freq in self.builder.multi_token_vocab.items()
            if token.isalpha() and len(token) > 1
        }
        bpe_tokens: Dict[Tuple[str, ...], int] = {
            tuple(token): freq for token, freq in bpe_vocab.items()
        }
        iteration: int = 0

        for _ in tqdm(range(num_merges), desc="BPE iterations", leave=False):
            iteration += 1
            pair_freqs: DefaultDict[Tuple[str, str], int] = defaultdict(int)
            for token_tuple, freq in bpe_tokens.items():
                for j in range(len(token_tuple) - 1):
                    pair = (token_tuple[j], token_tuple[j + 1])
                    pair_freqs[pair] += freq
            if not pair_freqs:
                logger.debug("No more pairs to merge in BPE.")
                break
            max_pair_freq = max(pair_freqs.values())
            merge_threshold = max(2, int(max_pair_freq * 0.1))
            most_frequent_pair = max(pair_freqs, key=lambda x: pair_freqs[x])
            if pair_freqs[most_frequent_pair] < merge_threshold:
                logger.debug(
                    "BPE stopping: Frequency %s of pair %s is below the merge threshold %s.",
                    pair_freqs[most_frequent_pair],
                    most_frequent_pair,
                    merge_threshold,
                )
                break
            logger.debug(
                "BPE iteration %d: Merging pair %s with frequency %s.",
                iteration,
                most_frequent_pair,
                pair_freqs[most_frequent_pair],
            )
            new_bpe_tokens: Dict[Tuple[str, ...], int] = {}
            for token_tuple, freq in bpe_tokens.items():
                new_token: List[str] = []
                j = 0
                while j < len(token_tuple):
                    if (
                        j < len(token_tuple) - 1
                        and (token_tuple[j], token_tuple[j + 1]) == most_frequent_pair
                    ):
                        new_token.append(token_tuple[j] + token_tuple[j + 1])
                        j += 2
                    else:
                        new_token.append(token_tuple[j])
                        j += 1
                new_bpe_tokens[tuple(new_token)] = (
                    new_bpe_tokens.get(tuple(new_token), 0) + freq
                )
            bpe_tokens = new_bpe_tokens

        for token_tuple, freq in bpe_tokens.items():
            merged_token: str = "".join(token_tuple)
            if merged_token not in self.builder.multi_token_vocab:
                self.builder._add_token(merged_token, freq)
                logger.info("BPE merged token added: %s", merged_token)
        logger.info("BPE subword merging process completed.")

    def apply_contextual_bpe(self, num_merges: int = 300) -> None:
        """
        Apply an enhanced BPE merging that considers contextual similarity.
        """
        logger.info("Starting contextual BPE merging process...")
        bpe_vocab: Dict[str, int] = {
            token: freq
            for token, freq in self.builder.multi_token_vocab.items()
            if token.isalpha() and len(token) > 1
        }
        bpe_tokens: Dict[Tuple[str, ...], int] = {
            tuple(token): freq for token, freq in bpe_vocab.items()
        }
        iteration: int = 0

        for _ in tqdm(range(num_merges), desc="Contextual BPE iterations", leave=False):
            iteration += 1
            pair_metrics: DefaultDict[Tuple[str, str], float] = defaultdict(float)
            for token_tuple, freq in bpe_tokens.items():
                for j in range(len(token_tuple) - 1):
                    pair = (token_tuple[j], token_tuple[j + 1])
                    metric: float = float(freq)
                    if self.builder.transformer_manager.model is not None:
                        import numpy as np

                        emb1 = self.builder.transformer_manager.embedding_cache.get(
                            token_tuple[j]
                        )
                        emb2 = self.builder.transformer_manager.embedding_cache.get(
                            token_tuple[j + 1]
                        )
                        if emb1 is None:
                            if self.builder.transformer_manager.tokenizer is None:
                                logger.warning(
                                    "Transformer tokenizer is not initialized; skipping contextual BPE for pair: %s",
                                    pair,
                                )
                            else:
                                inputs = self.builder.transformer_manager.tokenizer(
                                    token_tuple[j], return_tensors="pt"
                                )
                                with torch.no_grad():
                                    outputs = self.builder.transformer_manager.model(
                                        **inputs
                                    )
                                emb1 = (
                                    outputs.last_hidden_state.mean(dim=1)
                                    .squeeze()
                                    .tolist()
                                )
                                self.builder.transformer_manager.embedding_cache[token_tuple[j]] = emb1  # type: ignore
                        if emb2 is None:
                            if self.builder.transformer_manager.tokenizer is None:
                                logger.warning(
                                    "Transformer tokenizer is not initialized; skipping contextual BPE for pair: %s",
                                    pair,
                                )
                            else:
                                inputs = self.builder.transformer_manager.tokenizer(
                                    token_tuple[j + 1], return_tensors="pt"
                                )
                                with torch.no_grad():
                                    outputs = self.builder.transformer_manager.model(
                                        **inputs
                                    )
                                emb2 = (
                                    outputs.last_hidden_state.mean(dim=1)
                                    .squeeze()
                                    .tolist()
                                )
                                self.builder.transformer_manager.embedding_cache[token_tuple[j + 1]] = emb2  # type: ignore
                        emb1_np = np.array(emb1)
                        emb2_np = np.array(emb2)
                        cosine_sim = np.dot(emb1_np, emb2_np) / (
                            np.linalg.norm(emb1_np) * np.linalg.norm(emb2_np) + 1e-9
                        )
                        metric *= cosine_sim
                    pair_metrics[pair] += metric
            if not pair_metrics:
                logger.debug("No more pairs to merge in contextual BPE.")
                break
            max_metric = max(pair_metrics.values())
            merge_threshold = max(2.0, max_metric * 0.1)
            most_promising_pair = max(pair_metrics, key=lambda x: pair_metrics[x])
            if pair_metrics[most_promising_pair] < merge_threshold:
                logger.debug(
                    "Contextual BPE stopping: Metric %.2f for pair %s is below threshold %.2f.",
                    pair_metrics[most_promising_pair],
                    most_promising_pair,
                    merge_threshold,
                )
                break
            logger.debug(
                "Contextual BPE iteration %d: Merging pair %s with metric %.2f.",
                iteration,
                most_promising_pair,
                pair_metrics[most_promising_pair],
            )
            new_bpe_tokens: Dict[Tuple[str, ...], int] = {}
            for token_tuple, freq in bpe_tokens.items():
                new_token: List[str] = []
                j = 0
                while j < len(token_tuple):
                    if (
                        j < len(token_tuple) - 1
                        and (token_tuple[j], token_tuple[j + 1]) == most_promising_pair
                    ):
                        new_token.append(token_tuple[j] + token_tuple[j + 1])
                        j += 2
                    else:
                        new_token.append(token_tuple[j])
                        j += 1
                new_bpe_tokens[tuple(new_token)] = (
                    new_bpe_tokens.get(tuple(new_token), 0) + freq
                )
            bpe_tokens = new_bpe_tokens

        for token_tuple, freq in bpe_tokens.items():
            merged_token: str = "".join(token_tuple)
            if merged_token not in self.builder.multi_token_vocab:
                self.builder._add_token(merged_token, freq)
                logger.info("Contextual BPE merged token added: %s", merged_token)
        logger.info("Contextual BPE merging process completed.")

    def apply_sentencepiece(self) -> None:
        """
        Train and apply a SentencePiece model on the aggregated dynamic corpus.
        """
        corpus_file = str(self.builder.corpus_manager.dynamic_corpus_file)
        logger.info(
            "Training SentencePiece model on dynamic corpus from %s...", corpus_file
        )
        model_prefix = "spm_model_dynamic"
        try:
            spm.SentencePieceTrainer.train(  # type: ignore
                input=corpus_file,
                model_prefix=model_prefix,
                vocab_size=32000,
                character_coverage=1.0,
                model_type="bpe",
            )
        except Exception as e:
            logger.error("Error training SentencePiece model: %s", e)
            return
        sp = spm.SentencePieceProcessor()  # type: ignore
        sp.load(f"{model_prefix}.model")  # type: ignore
        sp_vocab = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]  # type: ignore
        for token in sp_vocab:
            if (
                token not in self.builder.multi_token_vocab
                and token.isalpha()
                and len(token) > 3
            ):
                self.builder._add_token(token, 1)
                logger.info("SentencePiece merged token added: %s", token)

    def apply_unigram_lm(self, num_merges: int = 500) -> None:
        """
        Train and apply a Unigram LM model on the aggregated dynamic corpus.
        """
        corpus_file = str(self.builder.corpus_manager.dynamic_corpus_file)
        logger.info(
            "Training Unigram LM model on dynamic corpus from %s...", corpus_file
        )
        model_prefix = "unigram_model_dynamic"
        try:
            spm.SentencePieceTrainer.train(  # type: ignore
                input=corpus_file,
                model_prefix=model_prefix,
                vocab_size=32000,
                character_coverage=1.0,
                model_type="unigram",
            )
        except Exception as e:
            logger.error("Error training Unigram LM model: %s", e)
            return
        sp = spm.SentencePieceProcessor()  # type: ignore
        sp.load(f"{model_prefix}.model")  # type: ignore
        sp_vocab = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]  # type: ignore
        for token in sp_vocab:
            if (
                token not in self.builder.multi_token_vocab
                and token.isalpha()
                and len(token) > 4
            ):
                self.builder._add_token(token, 1)
                logger.info("Unigram LM merged token added: %s", token)

    def apply_advanced_merging(self) -> None:
        """
        Apply a layered merging process combining BPE, contextual BPE, SentencePiece, and Unigram LM.
        """
        logger.info("Starting advanced merging processes...")
        if self.builder.use_advanced:
            self.builder._ensure_nltk_resources()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                futures.append(
                    executor.submit(self.builder.enhance_vocabulary_with_wordnet)
                )
                futures.append(
                    executor.submit(self.builder.enhance_vocabulary_with_nltk_words)
                )
                futures.append(
                    executor.submit(self.builder.enhance_vocabulary_with_wiktionary)
                )
                concurrent.futures.wait(futures)
        self.apply_bpe(num_merges=500)
        self.apply_contextual_bpe(num_merges=300)
        self.apply_sentencepiece()
        self.apply_unigram_lm()
        logger.info("Advanced merging processes completed.")


# ---------------------------------------------------------------------------
# Modular Component: Transformer Manager for embedding model operations.
class TransformerManager:
    def __init__(
        self,
        transformer_tokenizer: Any,
        transformer_model: Any,
        device: torch.device,
        embedding_cache: LRUCache,
    ):
        self.tokenizer = transformer_tokenizer
        self.model = transformer_model
        self.device = device
        self.embedding_cache = embedding_cache

    def load_model(self, model_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.train()  # Switch to training mode
        self.model.to(self.device)
        logger.info("Transformer embedding model loaded: %s", model_name)

    def async_train(self, inputs, dataset) -> None:
        trainer = Trainer(
            model=self.model,
            args=DEFAULT_TRAINING_ARGS,
            train_dataset=dataset,
        )
        trainer.train()
        logger.info("Transformer embedding model fine-tuned on dynamic corpus.")


# ---------------------------------------------------------------------------
# Modular Component: Corpus Manager for dynamic corpus updates.
class CorpusManager:
    def __init__(self, corpus_file: Path):
        self.dynamic_corpus: List[str] = []
        self.dynamic_corpus_file: Path = corpus_file
        self.lock = threading.Lock()

    def update(self, new_text: str) -> None:
        with self.lock:
            self.dynamic_corpus.append(new_text)
        try:
            with self.dynamic_corpus_file.open("a", encoding="utf-8") as f:
                f.write(new_text + "\n")
            logger.info("Dynamic corpus updated with new text.")
        except Exception as e:
            logger.error("Error updating dynamic corpus file: %s", e)
            raise


# ---------------------------------------------------------------------------
# Modular Component: AutoSaver for periodic saving of vocabulary.
class AutoSaver:
    def __init__(
        self, builder: VocabularyBuilder, auto_save_interval: int, auto_save_file: Path
    ):
        self.builder = builder
        self.auto_save_interval = auto_save_interval
        self.auto_save_file = auto_save_file
        self.shutdown_event = threading.Event()
        self.thread = threading.Thread(target=self._auto_save_loop, daemon=True)
        self.thread.start()

    def _auto_save_loop(self) -> None:
        while not self.shutdown_event.is_set():
            time.sleep(self.auto_save_interval)
            try:
                self.builder.save_vocab(str(self.auto_save_file))
                logger.info("Auto-saved vocabulary to %s", self.auto_save_file)
            except Exception as e:
                logger.error("Auto-save failed: %s", e)

    def stop(self) -> None:
        self.shutdown_event.set()
        self.thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Module‑level training configuration for transformer fine‑tuning.
DEFAULT_TRAINING_ARGS: TrainingArguments = TrainingArguments(  # type: ignore
    output_dir="./transformer_finetune",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    logging_steps=10,
    logging_dir="./logs",
    disable_tqdm=True,
)

# Set up module‑level logger with a stylish format.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Use INFO or WARNING in production.
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
)
logger.addHandler(_stream_handler)


# ---------------------------------------------------------------------------
# Main Class: VocabularyBuilder coordinating all components.
class VocabularyBuilder:
    """
    Constructs and manages a multi-token vocabulary using n-gram frequency analysis,
    enhanced by advanced NLP resources, multi-source linguistic metadata, and layered
    subword merging.
    """

    def __init__(
        self,
        tokenizer: CharTokenizer,
        min_count: int = 5,
        max_ngram: int = 5,
        mode: str = "character",
        lowercase: bool = True,
        use_advanced: bool = True,
    ) -> None:
        """
        Initialize the VocabularyBuilder.

        Args:
            tokenizer: An instance of CharTokenizer.
            min_count: Minimum frequency threshold for including n-grams.
            max_ngram: Maximum n-gram size to consider.
            mode: "character" for character-level extraction or "word" for NLP-based word extraction.
                  When mode is "word", spaCy will be used provided its model can be loaded.
            lowercase: Whether to convert text to lowercase before processing.
            use_advanced: Enable integration with additional linguistic resources and advanced merging.
        """
        self.tokenizer = tokenizer
        self.min_count = min_count
        self.max_ngram = max_ngram
        self.mode = mode
        self.lowercase = lowercase
        self.use_advanced = use_advanced
        self.ngram_counts: Counter[str] = Counter()
        self.multi_token_vocab: Dict[str, int] = {}
        # Store additional metadata for tokens.
        self.token_metadata: Dict[str, Dict[str, Any]] = {}

        # Initialize spaCy if in word mode.
        self.nlp: Optional[Language] = None
        if self.mode == "word":
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.debug(
                    "spaCy language model 'en_core_web_sm' loaded successfully."
                )
            except Exception as e:
                logger.error("Failed to load spaCy language model: %s", e)
                raise RuntimeError(
                    "spaCy must be installed and 'en_core_web_sm' available."
                ) from e

        # Initialize Trie for efficient greedy matching.
        self.trie = Trie()

        # Initialize caches.
        self.tokenize_cache = LRUCache(maxsize=4096)
        self.ngram_extraction_cache = LRUCache(maxsize=4096)

        # Create modular components.
        self.text_processor = TextProcessor(
            self.mode, self.lowercase, self.nlp, self.tokenize_cache
        )
        self.ngram_extractor = NGramExtractor(
            self.max_ngram, self.ngram_extraction_cache
        )

        # Initialize transformer manager.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer_manager = TransformerManager(
            transformer_tokenizer=AutoTokenizer.from_pretrained(
                "distilbert-base-uncased"
            ),
            transformer_model=AutoModel.from_pretrained("distilbert-base-uncased"),
            device=device,
            embedding_cache=LRUCache(maxsize=4096),
        )
        self.transformer_manager.model.train()  # Switch to training mode
        self.transformer_manager.model.to(device)
        logger.info("Transformer embedding model loaded: distilbert-base-uncased")

        # Metadata populator.
        self.metadata_populator = MetadataPopulator(
            nlp=self.nlp,
            transformer_model=self.transformer_manager.model,
            transformer_tokenizer=self.transformer_manager.tokenizer,
            embedding_cache=self.transformer_manager.embedding_cache,
        )

        # Initialize corpus manager.
        self.corpus_manager = CorpusManager(
            Path("~/Development/dynamic_corpus.txt").expanduser()
        )

        # Executor for concurrent tasks.
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

        # Set up auto-saving.
        self.auto_save_interval = 60  # seconds
        self.auto_save_file = Path("vocab_autosave.pkl")
        self.auto_saver = AutoSaver(self, self.auto_save_interval, self.auto_save_file)

        # Subword merger.
        self.subword_merger = SubwordMerger(self)

    def _add_token(self, token: str, count: int) -> None:
        """
        Helper method to add a token to the vocabulary, update the Trie, add it to the tokenizer,
        and populate additional metadata.
        """
        self.multi_token_vocab[token] = count
        self.trie.insert(token)
        if callable(getattr(self.tokenizer, "add_token", None)):
            self.tokenizer.add_token(token)
        if self.use_advanced:
            self.token_metadata[token] = self.metadata_populator.populate(token)

    def _ensure_nltk_resources(self) -> None:
        """
        Make sure that the required NLTK resources are downloaded and available.
        """
        for resource in ["wordnet", "words", "punkt"]:
            try:
                nltk.data.find(f"corpora/{resource}")
            except LookupError:
                logger.info("Downloading missing NLTK resource: %s", resource)
                nltk.download(resource)

    def enhance_vocabulary_with_wordnet(self) -> None:
        """
        Enhance the vocabulary by adding WordNet lemma names for each alphabetic token.
        """
        logger.info("Enhancing vocabulary with WordNet...")
        for token in list(self.multi_token_vocab.keys()):
            if token.isalpha():
                synsets = wn.synsets(token)
                for syn in synsets:
                    if syn is None:
                        continue
                    for lemma in syn.lemma_names():
                        lemma_lower = lemma.lower()
                        if lemma_lower not in self.multi_token_vocab:
                            self._add_token(lemma_lower, 1)
                            logger.info(
                                "Enhanced vocabulary with WordNet lemma: %s",
                                lemma_lower,
                            )

    def enhance_vocabulary_with_nltk_words(self) -> None:
        """
        Enhance the vocabulary using the nltk.words corpus by adding common English words.
        """
        try:
            from nltk.corpus import words as nltk_words
        except ImportError:
            logger.error("NLTK words corpus is not available.")
            return
        logger.info("Enhancing vocabulary with NLTK words dictionary...")
        word_list = set(w.lower() for w in nltk_words.words())
        for word in word_list:
            if word not in self.multi_token_vocab:
                self._add_token(word, 1)
        logger.info("Vocabulary successfully enhanced with NLTK words.")

    def enhance_vocabulary_with_wiktionary(self) -> None:
        """
        Enhance the vocabulary using simulated Wiktionary data.
        In a real implementation, this would query Wiktionary's API or parse Wiktionary dumps.
        """
        logger.info("Enhancing vocabulary with Wiktionary data (simulated)...")
        for token in list(self.multi_token_vocab.keys()):
            if token.isalpha() and token not in self.token_metadata.get(token, {}):
                self.metadata_populator.populate(token, source="wiktionary")
                logger.info(
                    "Enhanced vocabulary with Wiktionary data for token: %s", token
                )

    def apply_bpe(self, num_merges: int = 500) -> None:
        """
        Delegate BPE subword merging.
        """
        self.subword_merger.apply_bpe(num_merges)

    def apply_contextual_bpe(self, num_merges: int = 300) -> None:
        """
        Delegate contextual BPE merging.
        """
        self.subword_merger.apply_contextual_bpe(num_merges)

    def apply_sentencepiece(self) -> None:
        """
        Delegate SentencePiece merging.
        """
        self.subword_merger.apply_sentencepiece()

    def apply_unigram_lm(self, num_merges: int = 500) -> None:
        """
        Delegate Unigram LM merging.
        """
        self.subword_merger.apply_unigram_lm(num_merges)

    def apply_advanced_merging(self) -> None:
        """
        Apply a layered merging process combining BPE, contextual BPE, SentencePiece, and Unigram LM.
        """
        logger.info("Starting advanced merging processes...")
        if self.use_advanced:
            self._ensure_nltk_resources()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                futures.append(executor.submit(self.enhance_vocabulary_with_wordnet))
                futures.append(executor.submit(self.enhance_vocabulary_with_nltk_words))
                futures.append(executor.submit(self.enhance_vocabulary_with_wiktionary))
                concurrent.futures.wait(futures)
        self.apply_bpe(num_merges=500)
        self.apply_contextual_bpe(num_merges=300)
        self.apply_sentencepiece()
        self.apply_unigram_lm()
        logger.info("Advanced merging processes completed.")

    def _process_text(self, text: str) -> Counter[str]:
        """
        Process a single text sample to extract n-gram counts.
        """
        return self.ngram_extractor.process_text(text, self.text_processor)

    def build_from_corpus(self, corpus: Iterable[str]) -> None:
        """
        Build the multi-token vocabulary using n-gram frequency counts from the provided corpus.
        Processes the corpus using parallel processing and streaming with progress feedback.
        """
        logger.info("Starting to build vocabulary from corpus...")
        corpus_list: List[str] = list(corpus)
        total_texts: int = len(corpus_list)
        if total_texts == 0:
            raise ValueError("The provided corpus is empty; cannot build vocabulary.")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(
                tqdm(
                    executor.map(self._process_text, corpus_list),
                    total=total_texts,
                    desc="Building vocabulary...",
                )
            )
        for local_counter in results:
            self.ngram_counts.update(local_counter)
        for ngram, count in self.ngram_counts.items():
            if count >= self.min_count:
                self._add_token(ngram, count)
                logger.info(
                    "Added n-gram to vocabulary: %r with count %s", ngram, count
                )
        if self.use_advanced:
            self.apply_advanced_merging()
        logger.info(
            "Vocabulary building complete. Total texts processed: %s.", total_texts
        )

    def learn_in_real_time(self, text: str) -> None:
        """
        Incrementally update the vocabulary by processing a new text sample in real time.
        Detailed progress feedback is provided during processing.
        """
        logger.info("Starting real-time vocabulary learning...")
        original_text: str = text
        proc_text: str = self.text_processor.normalize(text)
        if self.mode == "word" and self.nlp:
            doc: Optional[Doc] = self.text_processor.get_doc(proc_text)
            if doc is not None:
                word_tokens = [token.text for token in doc if not token.is_space]
            else:
                logger.warning(
                    "spaCy tokenization error during real-time learning; falling back to simple split."
                )
                word_tokens = proc_text.split()
            char_tokens = list(proc_text)
            for tokens, joiner in ((word_tokens, " "), (char_tokens, "")):
                token_count = len(tokens)
                for ngram_size in range(2, self.max_ngram + 1):
                    if token_count >= ngram_size:
                        for i in range(token_count - ngram_size + 1):
                            ngram = joiner.join(tokens[i : i + ngram_size])
                            self.ngram_counts[ngram] += 1
                            if self.ngram_counts[ngram] >= self.min_count:
                                if ngram not in self.multi_token_vocab:
                                    self._add_token(ngram, self.ngram_counts[ngram])
                                    logger.info(
                                        "Real-time learning: Added new %s n-gram %r with count %s",
                                        "word" if joiner == " " else "char",
                                        ngram,
                                        self.ngram_counts[ngram],
                                    )
                                else:
                                    self.multi_token_vocab[ngram] = self.ngram_counts[
                                        ngram
                                    ]
        else:
            tokens = list(proc_text)
            token_count = len(tokens)
            for ngram_size in range(2, self.max_ngram + 1):
                if token_count >= ngram_size:
                    for i in range(token_count - ngram_size + 1):
                        ngram = "".join(tokens[i : i + ngram_size])
                        self.ngram_counts[ngram] += 1
                        if self.ngram_counts[ngram] >= self.min_count:
                            if ngram not in self.multi_token_vocab:
                                self._add_token(ngram, self.ngram_counts[ngram])
                                logger.info(
                                    "Real-time learning: Added new n-gram %r with count %s",
                                    ngram,
                                    self.ngram_counts[ngram],
                                )
                            else:
                                self.multi_token_vocab[ngram] = self.ngram_counts[ngram]
        logger.info("Real-time learning update complete for text: %r", original_text)
        self.executor.submit(self.corpus_manager.update, original_text)

    def generate_report(self) -> str:
        """
        Generate a comprehensive report summarizing vocabulary statistics.
        """
        total_unique_ngrams: int = len(self.ngram_counts)
        vocab_size: int = len(self.multi_token_vocab)
        report_lines: List[str] = []
        report_lines.append("========== Vocabulary Report ==========")
        report_lines.append(f"Total Unique N-grams Counted: {total_unique_ngrams}")
        report_lines.append(
            f"Vocabulary Size (n-grams meeting min_count): {vocab_size}"
        )
        average_frequency: float = (
            (sum(self.multi_token_vocab.values()) / vocab_size)
            if vocab_size > 0
            else 0.0
        )
        report_lines.append(
            f"Average Frequency of Vocabulary Tokens: {average_frequency:.2f}"
        )
        report_lines.append("Top 10 Tokens by Frequency:")
        top_tokens = sorted(
            self.multi_token_vocab.items(), key=lambda x: (-x[1], x[0])
        )[:10]
        for token, count in top_tokens:
            report_lines.append(f"    {token!r}: {count}")
        report_lines.append("=======================================")
        return "\n".join(report_lines)

    def retokenize(self, text: str) -> List[str]:
        """
        Retokenize the given text using the multi-token vocabulary.
        Uses spaCy for initial tokenization in word mode,
        and falls back to a Trie-based search in character mode.
        """
        original_text: str = text
        proc_text: str = self.text_processor.normalize(text)
        retok: List[str] = []
        if self.mode == "word" and self.nlp:
            doc: Optional[Doc] = self.text_processor.get_doc(proc_text)
            if doc is None:
                logger.error("spaCy tokenization failed; falling back to simple split.")
                doc = cast(
                    Doc,
                    [
                        type(
                            "DummyToken",
                            (object,),
                            {"text": t, "is_space": t.isspace()},
                        )()
                        for t in proc_text.split(" ")
                    ],
                )
            tokens = list(doc)
            i: int = 0
            while i < len(tokens):
                if tokens[i].is_space:
                    retok.append(tokens[i].text)
                    i += 1
                else:
                    non_space_tokens: List[str] = []
                    while i < len(tokens) and not tokens[i].is_space:
                        non_space_tokens.append(tokens[i].text)
                        i += 1
                    j: int = 0
                    while j < len(non_space_tokens):
                        found: Optional[str] = None
                        for size in range(
                            min(self.max_ngram, len(non_space_tokens) - j), 0, -1
                        ):
                            candidate = " ".join(non_space_tokens[j : j + size])
                            if candidate in self.multi_token_vocab:
                                found = candidate
                                logger.debug(
                                    "Matched multi-token sequence %r starting at index %d in non-space token block.",
                                    candidate,
                                    j,
                                )
                                break
                        if found:
                            retok.append(found)
                            j += len(found.split())
                        else:
                            fallback = self.trie.search_longest(non_space_tokens[j], 0)
                            if fallback:
                                retok.append(fallback)
                                j += len(fallback)
                            else:
                                retok.append(non_space_tokens[j])
                                j += 1
        else:
            i = 0
            while i < len(proc_text):
                token = self.trie.search_longest(proc_text, i)
                if token:
                    retok.append(token)
                    logger.debug(
                        "Matched multi-character sequence %r at index %d.", token, i
                    )
                    i += len(token)
                else:
                    retok.append(proc_text[i])
                    i += 1
        logger.info("Retokenized %r to %s", original_text, retok)
        return retok

    def save_vocab(self, filename: str) -> None:
        """
        Persist the vocabulary (multi-token vocabulary, n-gram counts, and token metadata) to disk.
        Uses atomic writing to ensure interrupt resistance.
        """
        tmp_filename = filename + ".tmp"
        try:
            with open(tmp_filename, "wb") as f:
                pickle.dump(
                    (self.multi_token_vocab, self.ngram_counts, self.token_metadata), f
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_filename, filename)
            logger.info("Vocabulary successfully saved to %s", filename)
        except Exception as e:
            logger.error("Error saving vocabulary: %s", e)
            raise

    def load_vocab(self, filename: str) -> None:
        """
        Load the vocabulary from disk and rebuild the Trie.
        """
        try:
            with open(filename, "rb") as f:
                self.multi_token_vocab, self.ngram_counts, self.token_metadata = (
                    pickle.load(f)
                )
            logger.info("Vocabulary loaded from %s", filename)
            self.trie = Trie()
            for token in self.multi_token_vocab.keys():
                self.trie.insert(token)
            logger.debug("Trie successfully rebuilt from loaded vocabulary.")
        except Exception as e:
            logger.error("Error loading vocabulary: %s", e)
            raise

    def update_corpus(self, additional_corpus: Iterable[str]) -> None:
        """
        Update the current vocabulary using additional text data.
        """
        logger.info("Updating vocabulary with new corpus data...")
        self.build_from_corpus(additional_corpus)

    def get_vocabulary(self) -> Dict[str, int]:
        """
        Retrieve the current multi-token vocabulary.
        """
        return dict(self.multi_token_vocab)

    def get_ngram_counts(self) -> Counter[str]:
        """
        Get the complete n-gram count statistics (prior to min_count filtering).
        """
        return self.ngram_counts

    def _async_transformer_train(self, inputs, dataset) -> None:
        assert (
            self.transformer_manager.model is not None
        ), "Transformer model must be initialized before training."
        trainer = Trainer(
            model=self.transformer_manager.model,
            args=DEFAULT_TRAINING_ARGS,
            train_dataset=dataset,
        )
        trainer.train()
        logger.info("Transformer embedding model fine-tuned on dynamic corpus.")

    def load_transformer_embedding_model(
        self, model_name: str = "distilbert-base-uncased"
    ) -> None:
        """
        Load and continuously fine-tune a transformer-based embedding model for context-sensitive embeddings.
        """
        self.transformer_manager.load_model(model_name)
        logger.info("Transformer embedding model loaded: %s", model_name)
        if hasattr(self, "corpus_manager") and self.corpus_manager.dynamic_corpus:
            texts = self.corpus_manager.dynamic_corpus
            inputs = self.transformer_manager.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            )
            dataset = TensorDataset(torch.tensor(inputs["input_ids"]))
            self.executor.submit(self._async_transformer_train, inputs, dataset)

    def update_dynamic_corpus(self, new_text: str) -> None:
        """
        Continuously update the dynamic corpus with new textual data.
        """
        self.corpus_manager.update(new_text)

    def shutdown(self) -> None:
        """
        Signal shutdown event and clean up resources such as thread pool executor and autosave thread.
        """
        logger.info("Shutting down VocabularyBuilder...")
        self.auto_saver.stop()
        self.executor.shutdown(wait=True)
        logger.info("VocabularyBuilder shut down successfully.")


# ---------------------------------------------------------------------------
def demo_vocabulary_builder() -> None:
    """
    Demonstrate the advanced capabilities of the VocabularyBuilder.
    This demo builds a rich multi-token vocabulary from a diverse corpus,
    leverages advanced merging (WordNet, Wiktionary, NLTK, transformer embeddings),
    and provides an interactive session for real-time vocabulary learning.
    """
    logger.info("=== Welcome to the Advanced Vocabulary Builder Demo ===")
    logger.info(
        "Initializing an advanced demonstration of vocabulary construction and NLP integration..."
    )

    # Define a rich, illustrative corpus with diverse linguistic content.
    corpus: List[str] = [
        "In a far-off kingdom, words and definitions dance in the realm of linguistic beauty.",
        "The quick brown fox jumps over the lazy dog with expressive lexicon and vivid imagery.",
        "Exploring advanced language processing with spaCy, NLTK, and Wiktionary enriches our vocabulary.",
        "Every token holds a story: a meaning from WordNet, morphological insights, and even contextual embeddings.",
        "Artificial intelligence and natural language understanding converge in this elaborate demonstration.",
        "Transformers capture nuanced contextual embeddings, enabling powerful language analysis.",
    ]

    # Initialize the custom tokenizer.
    tokenizer: CharTokenizer = CharTokenizer(
        normalization_form="NFC",
        unicode_strategy="extensive",
        category_profile="all",
        sort_mode="unicode",
        dynamic_rebuild=True,
        persistence_prefix=None,
    )

    # Determine processing mode.
    mode: str = "word" if spacy.util.is_package("en_core_web_sm") else "character"
    logger.info(f"Operating in {mode} mode.")

    # Initialize the VocabularyBuilder with advanced merging enabled.
    vocab_builder: VocabularyBuilder = VocabularyBuilder(
        tokenizer, min_count=1, max_ngram=3, mode=mode, use_advanced=True
    )

    # Load and fine-tune the transformer-based embedding model for context-sensitive embeddings.
    logger.info(
        "Loading transformer-based embedding model for advanced contextual embeddings..."
    )
    try:
        vocab_builder.load_transformer_embedding_model("distilbert-base-uncased")
        logger.info("Transformer model loaded and fine-tuning initiated.")
    except Exception as e:
        logger.error("Error loading transformer model: %s", e)

    # Build the vocabulary from the rich corpus.
    logger.info("Building vocabulary from corpus...")
    vocab_builder.build_from_corpus(corpus)

    # Display the full multi-token vocabulary with frequency counts.
    print("\n======== Advanced Multi-token Vocabulary ========")
    for token, count in sorted(
        vocab_builder.get_vocabulary().items(), key=lambda x: (-x[1], x[0])
    ):
        print(f"  {token!r:20s}: {count}")

    # Demonstrate retokenization of a sample text.
    sample_text: str = (
        "Transformers and linguistic analysis combine for deep language understanding."
    )
    print("\n======== Retokenization Demo ========")
    retokenized = vocab_builder.retokenize(sample_text)
    print(f"Original Text : {sample_text}")
    print(f"Retokenized   : {retokenized}")

    # Begin interactive real-time vocabulary learning session.
    print("\n======== Interactive Real-Time Vocabulary Learning ========")
    print("Enter new sentences to update vocabulary in real time.")
    print("Type 'status' to view current vocabulary statistics, or 'exit' to finish.\n")
    while True:
        try:
            user_input: str = input("Your input > ").strip()
            if user_input.lower() == "exit":
                break
            if user_input.lower() == "status":
                print("\n----- Current Vocabulary Report -----")
                print(vocab_builder.generate_report())
                print("--------------------------------------\n")
                continue
            if not user_input:
                continue
            previous_vocab_size = len(vocab_builder.get_vocabulary())
            future = vocab_builder.executor.submit(
                vocab_builder.learn_in_real_time, user_input
            )
            future.result()  # Ensure completion before proceeding.
            new_vocab_size = len(vocab_builder.get_vocabulary())
            added_tokens = new_vocab_size - previous_vocab_size
            retokenized_input: List[str] = vocab_builder.retokenize(user_input)
            print("\n--- Input Analysis ---")
            print(f"Original      : {user_input}")
            print(f"Retokenized   : {retokenized_input}")
            print(
                f"New tokens added: {added_tokens}. Total vocabulary size: {new_vocab_size}\n"
            )
        except Exception as e:
            logger.error("Error during interactive learning: %s", e)
            print("An error occurred while processing your input. Please try again.")

    # Print final vocabulary report and shut down.
    print("\n======== Final Comprehensive Vocabulary Report ========")
    print(vocab_builder.generate_report())
    vocab_builder.shutdown()
    print("\n=== Advanced Vocabulary Builder Demo Complete ===")


if __name__ == "__main__":
    demo_vocabulary_builder()
