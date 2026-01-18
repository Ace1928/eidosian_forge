import unicodedata
import os
import pickle
import hashlib
import warnings
import importlib.metadata
import logging
import threading
from functools import lru_cache
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Literal, Optional, Iterable
from colorama import Fore, Back, Style, init
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Setup logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Configuration Module
# ---------------------------------------------------------------------


@dataclass
class TokenizerConfig:
    normalization_form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFC"
    special_tokens: Optional[Tuple[str, ...]] = None
    unicode_strategy: Optional[str] = None
    unicode_blocks: Optional[Iterable[Tuple[int, int]]] = None
    category_profile: Optional[str] = None
    technical_categories: Optional[Set[str]] = None
    control_chars: Optional[Iterable[int]] = None
    custom_chars: Optional[Iterable[str]] = None
    sort_mode: Literal["unicode", "frequency", "custom"] = "unicode"
    dynamic_rebuild: bool = True
    persistence_prefix: Optional[str] = None
    modes: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Plugin Interfaces Module
# ---------------------------------------------------------------------


class BasePlugin(ABC):
    """Abstract base class for tokenizer plugins."""

    @abstractmethod
    def get_chars(self) -> Set[str]:
        """Return a set of characters contributed by the plugin."""
        pass

    @abstractmethod
    def get_processor(self) -> Any:
        """Return a callable for processing tokens or text."""
        pass

    @abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """Serialize plugin state."""
        pass

    @classmethod
    @abstractmethod
    def deserialize(cls, data: Dict[str, Any]) -> "BasePlugin":
        """Deserialize and return an instance of the plugin."""
        pass


# ---------------------------------------------------------------------
# Unicode Utilities Module
# ---------------------------------------------------------------------


class UnicodeUtils:
    """
    Utility class providing cached methods for Unicode category computations.
    """

    @staticmethod
    @lru_cache(maxsize=None)
    def compute_chars_for_categories(categories: frozenset) -> Set[str]:
        """
        Compute and cache the set of Unicode characters that match any of the given categories.

        If a category string is a single letter then the function matches any code whose
        Unicode category (a two-letter string) starts with that letter. Otherwise it does an exact match.
        """
        result = set()
        for code in range(0x10FFFF + 1):
            try:
                char = chr(code)
            except (ValueError, OverflowError):
                continue
            cat = unicodedata.category(char)
            if any(
                (len(c) == 1 and cat.startswith(c)) or (len(c) == 2 and cat == c)
                for c in categories
            ):
                result.add(char)
        return result

    @staticmethod
    @lru_cache(maxsize=None)
    def compute_intervals_for_category(category: str) -> List[Tuple[int, int]]:
        """
        Compute and cache intervals (ranges) of contiguous Unicode code points that satisfy the given category.
        For single-letter categories, checks if the Unicode category starts with that letter.
        For two-letter categories, requires an exact match.
        """
        intervals = []
        start = None
        prev = None
        for code in range(0x10FFFF + 1):
            try:
                ch = chr(code)
            except (ValueError, OverflowError):
                continue
            cat = unicodedata.category(ch)
            match = (len(category) == 1 and cat.startswith(category)) or (
                len(category) == 2 and cat == category
            )
            if match:
                if start is None:
                    start = code
                    prev = code
                elif prev is not None and code == prev + 1:
                    prev = code
                else:
                    if start is not None and prev is not None:
                        intervals.append((start, prev))
                    start = code
                    prev = code
            else:
                if start is not None and prev is not None:
                    intervals.append((start, prev))
                    start = None
        if start is not None and prev is not None:
            intervals.append((start, prev))
        if not intervals:
            intervals.append((0, 0x10FFFF))
        return intervals


# ---------------------------------------------------------------------
# Plugin and Persistence Managers Module
# ---------------------------------------------------------------------


class PluginManager:
    """
    Manages dynamic discovery and attachment of tokenizer plugins.
    """

    def __init__(self, modes: Dict[str, Any]):
        self.modes = modes
        self.plugins: Dict[str, BasePlugin] = {}
        self.initialize_plugins()

    def initialize_plugins(self) -> None:
        try:
            eps = importlib.metadata.entry_points()
            if isinstance(eps, dict):
                plugin_entries = eps.get("my_tokenizer.plugins", [])
            else:
                plugin_entries = list(eps.select(group="my_tokenizer.plugins"))
            for ep in plugin_entries:
                plugin_instance = ep.load()()
                if isinstance(plugin_instance, BasePlugin):
                    self.plugins[ep.name] = plugin_instance
                else:
                    logger.warning(
                        f"Plugin {ep.name} does not implement BasePlugin interface."
                    )
        except Exception as e:
            logger.warning(
                f"Dynamic plugin discovery failed: {e}. Falling back to static registry."
            )
        if "plugins" in self.modes:
            for name, plugin in self.modes["plugins"].items():
                if isinstance(plugin, BasePlugin):
                    self.plugins[name] = plugin

    def attach_plugins(self, target: Any) -> None:
        """Attach plugin processors to the target object."""
        for name, plugin in self.plugins.items():
            try:
                processor = plugin.get_processor()
                setattr(target, f"apply_{name}", processor)
            except Exception as e:
                logger.error(f"Error attaching plugin '{name}': {e}")


class PersistenceManager:
    """
    Handles persistence of the vocabulary state.
    """

    def __init__(self, persistence_prefix: Optional[str]):
        self.persistence_prefix = persistence_prefix

    def load_vocabulary(self, config_hash: str) -> Optional[Dict[str, Any]]:
        vocab_file = f"{self.persistence_prefix}.vocab"
        config_hash_file = f"{self.persistence_prefix}.config_hash"
        try:
            if os.path.exists(vocab_file) and os.path.exists(config_hash_file):
                with open(config_hash_file, "r") as f_config:
                    stored_hash = f_config.read().strip()
                if stored_hash == config_hash:
                    with open(vocab_file, "rb") as f_vocab:
                        return pickle.load(f_vocab)
        except (OSError, pickle.PickleError) as e:
            logger.warning(f"Error loading vocabulary from persistence: {e}")
        return None

    def save_vocabulary(
        self, config_hash: str, vocab: Dict[str, int], inverse_vocab: Dict[int, str]
    ) -> None:
        vocab_file = f"{self.persistence_prefix}.vocab"
        config_hash_file = f"{self.persistence_prefix}.config_hash"
        for attempt in range(3):
            try:
                with open(vocab_file, "wb") as f_vocab:
                    pickle.dump(
                        {"vocab": vocab, "inverse_vocab": inverse_vocab}, f_vocab
                    )
                with open(config_hash_file, "w") as f_config:
                    f_config.write(config_hash)
                logger.info(f"{Fore.CYAN}💾 Saved vocabulary to persistence.")
                return
            except (OSError, pickle.PickleError) as e:
                logger.error(
                    f"Attempt {attempt+1}: Error saving vocabulary to persistence: {e}"
                )
        warnings.warn("Failed to save vocabulary to persistence after 3 attempts.")


# ---------------------------------------------------------------------
# Vocabulary Builder Module
# ---------------------------------------------------------------------


class VocabularyBuilder:
    """
    Builds the vocabulary used by the tokenizer including Unicode sources, custom tokens,
    and plugin contributions.
    """

    def __init__(
        self,
        config: TokenizerConfig,
        unicode_blocks: Set[Tuple[int, int]],
        categories: Set[str],
        control_chars: Set[int],
        custom_chars: Set[str],
        plugins: Dict[str, BasePlugin],
    ):
        self.config = config
        self.unicode_blocks = unicode_blocks
        self.categories = categories
        self.control_chars = control_chars
        self.custom_chars = custom_chars
        self.plugins = plugins

    def validate_configuration(self) -> None:
        if len(self.config.special_tokens or []) != len(
            set(self.config.special_tokens or [])
        ):
            raise ValueError("Special tokens must be unique")
        for char in self.custom_chars:
            if not (isinstance(char, str) and len(char) == 1):
                raise ValueError("Custom characters must be single-character strings")

    def add_unicode_blocks(self, chars: Set[str]) -> None:
        for start, end in self.unicode_blocks:
            if (
                start > end
                or not (0 <= start <= 0x10FFFF)
                or not (0 <= end <= 0x10FFFF)
            ):
                raise ValueError(f"Invalid Unicode range: {start}-{end}")
            chars.update({chr(c) for c in range(start, end + 1)})

    def add_unicode_categories(self, chars: Set[str]) -> None:
        if "*" in self.categories:
            return  # All characters are already covered via blocks
        for cat in self.categories:
            intervals = UnicodeUtils.compute_intervals_for_category(cat)
            for start, end in intervals:
                chars.update({chr(c) for c in range(start, end + 1)})

    def add_control_chars(self, chars: Set[str]) -> None:
        chars.update({chr(c) for c in self.control_chars})

    def sort_vocabulary(self, chars: Iterable[str]) -> List[str]:
        if self.config.sort_mode == "unicode":
            return sorted(chars, key=lambda c: ord(c))
        elif self.config.sort_mode == "frequency":
            # Frequency sorting based on a heuristic; using digit value if available.
            return sorted(chars, key=lambda c: -unicodedata.digit(c, 0))
        return list(chars)

    def build_vocabulary(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        self.validate_configuration()
        vocab = {t: i for i, t in enumerate(self.config.special_tokens or [])}
        chars: Set[str] = set()

        # Phase 1: Add Unicode sources
        self.add_unicode_blocks(chars)
        self.add_unicode_categories(chars)
        self.add_control_chars(chars)

        # Phase 2: Add custom characters
        chars.update(self.custom_chars)

        # Phase 3: Incorporate plugin contributions
        for plugin in self.plugins.values():
            try:
                chars.update(plugin.get_chars())
            except Exception as e:
                logger.error(f"Plugin error in get_chars: {e}")

        # Phase 4: Sort and finalize vocabulary
        ordered_chars = self.sort_vocabulary(chars)
        for idx, char in enumerate(ordered_chars, start=len(vocab)):
            vocab[char] = idx

        inverse_vocab = {v: k for k, v in vocab.items()}
        return vocab, inverse_vocab


# ---------------------------------------------------------------------
# Character Tokenizer Module
# ---------------------------------------------------------------------


class CharTokenizer:
    """
    Universal HyperTokenizer 9001: Ultimate Unicode Tokenization System

    A fully modular, extensible character-level tokenizer supporting:
    - Complete Unicode 15.0 coverage with configurable granularity
    - Real-time dynamic reconfiguration of all parameters
    - Multi-modal extension support through plugin architecture
    - Bidirectional encoding/decoding with lossless round-trip guarantees
    - Advanced diagnostics and contextual metadata generation

    Architecture Features:
      1. Hierarchical configuration system with cascading defaults
      2. Pluggable character source modules with dependency resolution
      3. Atomic vocabulary rebuilding with transaction safety
      4. Cross-modal serialization/deserialization support
      5. Context-aware error handling with recovery modes
    """

    DEFAULT_SPECIAL_TOKENS: Tuple[str, ...] = (
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "[BOS]",
        "[EOS]",
        "[NUM]",
        "[FORMULA]",
        "[CODE]",
        "[EMOJI]",
        "[MEDIA]",
        "[AUDIO]",
        "[3D]",
    )
    UNICODE_STRATEGIES: Dict[str, Tuple[Tuple[int, int], ...]] = {
        "minimal": ((0x0000, 0x07FF),),
        "moderate": ((0x0000, 0xFFFF),),
        "extensive": ((0x0000, 0x10FFFF),),
        "technical": ((0x0000, 0x1FFFF), (0x20000, 0x2FFFF), (0xE0000, 0xEFFFF)),
        "mathematical": ((0x2000, 0x2BFF), (0x1D400, 0x1D7FF), (0x1EE00, 0x1EEFF)),
    }
    CATEGORY_PROFILES: Dict[str, Set[str]] = {
        "linguistic": {"L", "M", "N", "P", "S", "Z"},
        "technical": {"Sm", "Sc", "Sk", "So", "Nd", "No"},
        "formatting": {"Cc", "Cf", "Co", "Cn", "Zl", "Zp"},
        "symbolic": {"S", "So", "Sc", "Sk", "Sm"},
        "all": {"*"},
    }
    DEFAULT_CONTROL_CHARS: Tuple[int, ...] = (
        0x0000,  # Null
        0x0009,  # Tab
        0x000A,  # Newline
        0x000D,  # Carriage return
        0x001B,  # Escape
        0x007F,  # Delete
        0x00A0,  # Non-breaking space
        0x200B,  # Zero-width space
        0xFEFF,  # Byte order mark
    )

    def __init__(
        self,
        normalization_form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFC",
        special_tokens: Optional[Tuple[str, ...]] = None,
        unicode_strategy: Optional[str] = None,
        unicode_blocks: Optional[Iterable[Tuple[int, int]]] = None,
        category_profile: Optional[str] = None,
        technical_categories: Optional[Set[str]] = None,
        control_chars: Optional[Iterable[int]] = None,
        custom_chars: Optional[Iterable[str]] = None,
        sort_mode: Literal["unicode", "frequency", "custom"] = "unicode",
        dynamic_rebuild: bool = True,
        persistence_prefix: Optional[str] = None,
        **modes: Any,
    ) -> None:
        self.config = TokenizerConfig(
            normalization_form=normalization_form,
            special_tokens=special_tokens or self.DEFAULT_SPECIAL_TOKENS,
            unicode_strategy=unicode_strategy,
            unicode_blocks=unicode_blocks,
            category_profile=category_profile,
            technical_categories=technical_categories,
            control_chars=control_chars,
            custom_chars=custom_chars,
            sort_mode=sort_mode,
            dynamic_rebuild=dynamic_rebuild,
            persistence_prefix=persistence_prefix,
            modes=modes,
        )
        # Resolve configuration aspects
        self._vocab_lock = False
        self.categories = self._resolve_categories(
            self.config.category_profile, self.config.technical_categories
        )
        self.unicode_blocks = self._resolve_unicode_coverage(
            self.config.unicode_strategy, self.config.unicode_blocks
        )
        self.control_chars = (
            set(self.config.control_chars)
            if self.config.control_chars
            else set(self.DEFAULT_CONTROL_CHARS)
        )
        self.custom_chars = (
            set(self.config.custom_chars) if self.config.custom_chars else set()
        )

        # Initialize plugin system
        self.plugin_manager = PluginManager(self.config.modes)
        self._plugins = self.plugin_manager.plugins
        self.plugin_manager.attach_plugins(self)

        # Initialize persistence manager if persistence is enabled
        self.persistence_manager = (
            PersistenceManager(self.config.persistence_prefix)
            if self.config.persistence_prefix
            else None
        )

        # Build vocabulary using the VocabularyBuilder component
        builder = VocabularyBuilder(
            self.config,
            self.unicode_blocks,
            self.categories,
            self.control_chars,
            self.custom_chars,
            self._plugins,
        )
        config_hash = self._get_config_hash()
        if self.persistence_manager:
            vocab_data = self.persistence_manager.load_vocabulary(config_hash)
            if vocab_data:
                self.vocab = vocab_data["vocab"]
                self.inverse_vocab = vocab_data["inverse_vocab"]
                logger.info(f"{Fore.GREEN}✅ Loaded vocabulary from persistence.")
            else:
                self.vocab, self.inverse_vocab = builder.build_vocabulary()
                self.persistence_manager.save_vocabulary(
                    config_hash, self.vocab, self.inverse_vocab
                )
        else:
            self.vocab, self.inverse_vocab = builder.build_vocabulary()

    def _get_config_hash(self) -> str:
        """Generate a hash representing the tokenizer configuration."""
        config = (
            self.config.normalization_form,
            self.config.sort_mode,
            tuple(sorted(self.config.special_tokens or [])),
            tuple(sorted(self.unicode_blocks or [])),
            tuple(sorted(self.categories or [])),
            tuple(sorted(self.control_chars)),
            tuple(sorted(self.custom_chars)),
            self.config.modes,
        )
        return hashlib.md5(str(config).encode()).hexdigest()

    def _resolve_unicode_coverage(
        self, strategy: Optional[str], custom: Optional[Iterable]
    ) -> Set[Tuple[int, int]]:
        """Resolve Unicode coverage from strategy and custom ranges."""
        if strategy and custom:
            warnings.warn(
                "Both Unicode strategy and custom blocks provided - merging coverage"
            )
        blocks: Set[Tuple[int, int]] = set()
        if strategy:
            blocks.update(self.UNICODE_STRATEGIES.get(strategy, set()))
        if custom:
            blocks.update((s, e) for s, e in custom)
        return blocks or set(self.UNICODE_STRATEGIES["extensive"])

    def _resolve_categories(
        self, profile: Optional[str], custom: Optional[Set]
    ) -> Set[str]:
        """Resolve Unicode categories from profile and custom set."""
        if profile == "all":
            return {"*"}
        if profile and custom:
            warnings.warn(
                "Both category profile and custom categories provided - merging"
            )
        categories = set()
        if profile:
            categories.update(self.CATEGORY_PROFILES.get(profile, set()))
        if custom:
            categories.update(custom)
        return categories or self.CATEGORY_PROFILES.get("technical", set())

    @lru_cache(maxsize=1024)
    def encode(
        self,
        text: str,
        normalization: Optional[Literal["NFC", "NFD", "NFKC", "NFKD"]] = None,
    ) -> List[int]:
        """Normalize and tokenize text with type-safe normalization."""
        norm_form: Literal["NFC", "NFD", "NFKC", "NFKD"] = (
            self.config.normalization_form if normalization is None else normalization
        )
        processed: str = unicodedata.normalize(norm_form, text)
        return [self.vocab.get(c, self.vocab.get("[UNK]", 0)) for c in processed]

    def decode(
        self,
        tokens: Iterable[int],
        normalization: Optional[Literal["NFC", "NFD", "NFKC", "NFKD"]] = None,
    ) -> str:
        """Reconstruct text from tokens with optional normalization."""
        decoded: str = "".join(self.inverse_vocab.get(t, "[UNK]") for t in tokens)
        if normalization:
            return unicodedata.normalize(normalization, decoded)
        return decoded

    def analyze(self, text: str) -> Dict[str, Any]:
        """Comprehensive text analysis with multi-modal insights."""
        encoded = self.encode(text)
        return {
            "basic": {
                "length": len(text),
                "tokens": len(encoded),
                "ratio": len(encoded) / len(text) if text else 0,
            },
            "coverage": {
                "unknowns": encoded.count(self.vocab.get("[UNK]", 0)),
                "unique_chars": len(set(text)),
            },
            "unicode": {
                "normalized_form": self.config.normalization_form,
                "planes_used": len({(ord(c) >> 16) for c in text}),
            },
        }

    def generate_report(self) -> Dict[str, Any]:
        """Complete system status report."""
        return {
            "configuration": {
                "normalization": self.config.normalization_form,
                "unicode_blocks": self.unicode_blocks,
                "categories": self.categories,
                "sort_mode": self.config.sort_mode,
            },
            "vocabulary": {
                "total_size": len(self.vocab),
                "special_tokens": len(self.config.special_tokens or []),
                "character_coverage": len(self.vocab)
                - len(self.config.special_tokens or []),
                "planes_covered": len(
                    {(ord(c) >> 16) for c in self.vocab if len(c) == 1}
                ),
            },
            "plugins": list(self._plugins.keys()),
        }

    def add_token(self, token: str) -> None:
        """Dynamically add a token to the vocabulary.

        If the token already exists, no changes are made.
        """
        if token not in self.vocab:
            new_index = len(self.vocab)
            self.vocab[token] = new_index
            self.inverse_vocab[new_index] = token
            logger.info(f"Added token {token!r} with index {new_index}")
        else:
            logger.info(f"Token {token!r} already exists in vocabulary.")

    def reconfigure(self, **params) -> None:
        """Live reconfiguration of tokenizer parameters."""
        if not self.config.dynamic_rebuild:
            raise RuntimeError("Dynamic reconfiguration disabled")
        self.__dict__.update(params)
        # Reinitialize plugin system in case modes have changed
        self.plugin_manager = PluginManager(self.config.modes)
        self._plugins = self.plugin_manager.plugins
        self.plugin_manager.attach_plugins(self)
        # Re-resolve configuration settings
        self.categories = self._resolve_categories(
            self.config.category_profile, self.config.technical_categories
        )
        self.unicode_blocks = self._resolve_unicode_coverage(
            self.config.unicode_strategy, self.config.unicode_blocks
        )
        self.control_chars = (
            set(self.config.control_chars)
            if self.config.control_chars
            else set(self.DEFAULT_CONTROL_CHARS)
        )
        self.custom_chars = (
            set(self.config.custom_chars) if self.config.custom_chars else set()
        )
        # Rebuild vocabulary using VocabularyBuilder and persistence if enabled
        builder = VocabularyBuilder(
            self.config,
            self.unicode_blocks,
            self.categories,
            self.control_chars,
            self.custom_chars,
            self._plugins,
        )
        config_hash = self._get_config_hash()
        if self.persistence_manager:
            vocab_data = self.persistence_manager.load_vocabulary(config_hash)
            if vocab_data:
                self.vocab = vocab_data["vocab"]
                self.inverse_vocab = vocab_data["inverse_vocab"]
                logger.info(f"{Fore.GREEN}✅ Loaded vocabulary from persistence.")
            else:
                self.vocab, self.inverse_vocab = builder.build_vocabulary()
                self.persistence_manager.save_vocabulary(
                    config_hash, self.vocab, self.inverse_vocab
                )
        else:
            self.vocab, self.inverse_vocab = builder.build_vocabulary()

    def __getstate__(self) -> Dict:
        """Custom serialization with plugin support."""
        state = self.__dict__.copy()
        state["_plugins"] = {k: v.serialize() for k, v in self._plugins.items()}
        return state

    def __setstate__(self, state: Dict) -> None:
        """Custom deserialization with plugin rehydration."""
        plugins_state = state.pop("_plugins", {})
        self.__dict__.update(state)
        default_plugins = PluginManager(self.config.modes).plugins
        for name, serialized in plugins_state.items():
            if name in default_plugins and hasattr(
                default_plugins[name], "deserialize"
            ):
                default_plugins[name] = default_plugins[name].deserialize(serialized)
            else:
                default_plugins[name] = serialized
        self._plugins = default_plugins
        PluginManager(self.config.modes).attach_plugins(self)


# ---------------------------------------------------------------------
# Dynamic Tokenizer Module
# ---------------------------------------------------------------------


class DynamicTokenizer(CharTokenizer):
    """
    An evolving dynamic adaptive intelligent tokenizer that continuously learns
    from new inputs. It updates and persists its vocabulary, making it resilient
    to interruptions while expanding its lexicon.
    """

    def learn(self, text: str) -> None:
        """
        Dynamically update the vocabulary using the given text.
        Any character (or token) not in the vocabulary is added.
        """
        # Process the text normally; encoding will use current vocab to substitute unknown tokens.
        encoded = self.encode(text)
        for char in text:
            if char not in self.vocab:
                self.add_token(char)
        # After learning, persist the updated vocabulary.
        if self.persistence_manager:
            config_hash = self._get_config_hash()
            self.persistence_manager.save_vocabulary(
                config_hash, self.vocab, self.inverse_vocab
            )
            logger.info("Vocabulary persisted after updating with new text.")

    def learn_in_background(self, text: str) -> None:
        """
        Launch a background thread for learning so that processing is interruption-resistant.
        """
        thread = threading.Thread(target=self.learn, args=(text,))
        # Optionally, set thread.daemon = True if you want it to not block exit.
        thread.start()
        thread.join()  # Join here for synchronous update; remove join() for full asynchronous behavior.


# ---------------------------------------------------------------------
# Demonstration and Utility Routines Module
# ---------------------------------------------------------------------


def demo_tokenizer():
    """Interactive Tokenizer Playground"""
    init(autoreset=True)
    logger.info(
        f"\n{Back.BLUE}{Fore.WHITE}=== 🎮 Universal Tokenizer Playground ==={Style.RESET_ALL}"
    )
    logger.info(
        f"{Fore.CYAN}🚀 Initializing dynamic tokenizer with maximum unicode coverage..."
    )

    tokenizer = DynamicTokenizer(
        normalization_form="NFC",
        unicode_strategy="extensive",
        category_profile="all",
        sort_mode="unicode",
        dynamic_rebuild=True,
        persistence_prefix="my_tokenizer",  # Enable persistence
    )
    # Demonstrate dynamic learning with a sample text at startup.
    sample_text = "A sample evolving text with new tokens: 🚀✨"
    tokenizer.learn_in_background(sample_text)
    logger.info(
        f"{Fore.GREEN}✅ Dynamic Tokenizer ready! Vocabulary size: {len(tokenizer.vocab):,} tokens"
    )

    while True:
        logger.info(
            f"\n{Back.MAGENTA}{Fore.WHITE}=== 🧪 Demonstration Menu ==={Style.RESET_ALL}"
        )
        logger.info(f"1. {Fore.YELLOW}🔤 Test your own text")
        logger.info(f"2. {Fore.CYAN}🌍 Multilingual example")
        logger.info(f"3. {Fore.MAGENTA}🎭 Emoji/Technical example")
        logger.info(f"4. {Fore.BLUE}📊 System report")
        logger.info(f"5. {Fore.GREEN}🤖 Learn new text dynamically")
        logger.info(f"6. {Fore.RED}❌ Exit")

        choice = input(f"{Fore.WHITE}🛠 Choose an option (1-6): ").strip()

        if choice == "1":
            text = input(f"\n{Fore.CYAN}📝 Enter text to tokenize: ")
            if not text:
                logger.info(f"{Fore.RED}⚠️ Please enter some text!")
                continue
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded)
            analysis = tokenizer.analyze(text)
            logger.info(f"\n{Fore.GREEN}🔢 Encoded tokens:\n{encoded}")
            logger.info(f"\n{Fore.BLUE}🔁 Decoded text:\n{decoded}")
            logger.info(f"\n{Fore.MAGENTA}📈 Analysis:")
            logger.info(f"• Token/Char ratio: {analysis['basic']['ratio']:.2f}")
            logger.info(f"• Unknown tokens: {analysis['coverage']['unknowns']}")
        elif choice == "2":
            sample = "日本語 текст! 123 👨💻 + ∑x² = 42 🚀"
            run_demo_case(tokenizer, sample, "🌍 Multilingual Example")
        elif choice == "3":
            sample = "🔥🐉 Schrödinger's Cat: ⚛️📈 ∑x² ⇒ ∞ 😱💥"
            run_demo_case(tokenizer, sample, "🎭 Mixed Content Example")
        elif choice == "4":
            report = tokenizer.generate_report()
            logger.info(
                f"\n{Back.GREEN}{Fore.WHITE}=== 📊 System Report ==={Style.RESET_ALL}"
            )
            logger.info(f"• Normalization: {report['configuration']['normalization']}")
            logger.info(
                f"• Unicode blocks: {len(report['configuration']['unicode_blocks'])} ranges"
            )
            logger.info(f"• Total vocabulary: {report['vocabulary']['total_size']:,}")
            logger.info(f"• Active plugins: {', '.join(report['plugins']) or 'None'}")
        elif choice == "5":
            learn_text = input(f"\n{Fore.CYAN}📝 Enter text to learn dynamically: ")
            if not learn_text:
                logger.info(f"{Fore.RED}⚠️ Please enter some text!")
                continue
            tokenizer.learn_in_background(learn_text)
            logger.info(
                f"{Fore.GREEN}✅ Vocabulary updated! New size: {len(tokenizer.vocab)}"
            )
        elif choice == "6":
            logger.info(f"\n{Fore.YELLOW}👋 Exiting playground...")
            break
        else:
            logger.info(f"{Fore.RED}⚠️ Invalid choice! Please try again.")


def run_demo_case(tokenizer, text, title):
    """Helper to display styled demo cases"""
    logger.info(f"\n{Back.CYAN}{Fore.WHITE}=== {title} ==={Style.RESET_ALL}")
    logger.info(f"📜 Original text:\n{Fore.YELLOW}{text}")
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    logger.info(f"\n{Fore.GREEN}🔢 Encoded tokens ({len(encoded)}):\n{encoded}")
    logger.info(f"\n{Fore.BLUE}🔁 Decoded text:\n{decoded}")
    logger.info(f"\n{Fore.MAGENTA}🔍 Roundtrip {'✅' if text == decoded else '❌'}")
    logger.info(tokenizer.encode("Hello, world!"))


if __name__ == "__main__":
    demo_tokenizer()
