"""
Model interface for Eidosian Forge.

Provides standardized interfaces for loading and interacting with language models.
Optimized for CPU usage with optional GPU acceleration.
"""

import importlib.util
import logging
import threading
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional

from ..models import ModelConfig

logger = logging.getLogger(__name__)


class ModelInterface(ABC):
    """Abstract base class for language model interfaces."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate completion for a prompt.

        Args:
            prompt: Input text
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        """
        Generate streaming completion for a prompt.

        Args:
            prompt: Input text
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature

        Returns:
            Iterator of generated text chunks
        """
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text into token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        pass

    @abstractmethod
    def num_tokens(self, text: str) -> int:
        """
        Count the number of tokens in text.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        pass


class ModelManager:
    """
    Manages loading and interaction with language models.

    Supports multiple model types with consistent interface, prioritizing
    CPU efficiency and low memory usage.
    """

    # Supported model types and their import paths
    MODEL_TYPES = {
        "qwen": "agent_forge.models.qwen_model",
        "gemma": "agent_forge.models.gemma_model",
        "deepseek": "agent_forge.models.deepseek_model",
        "llamacpp": "agent_forge.models.llamacpp_model",
        "huggingface": "agent_forge.models.huggingface_model",
    }

    def __init__(self, model_config: ModelConfig):
        """
        Initialize model manager.

        Args:
            model_config: Configuration for the language model
        """
        self.config = model_config
        self.model: Optional[ModelInterface] = None
        self.model_loaded = False
        self.loading_lock = threading.Lock()

    def load_model(self) -> None:
        """
        Load the language model based on configuration.

        Dynamically imports the appropriate model implementation based on model_type.
        """
        if self.model_loaded:
            logger.debug("Model already loaded")
            return

        with self.loading_lock:
            if self.model_loaded:  # Double-check inside lock
                return

            model_type = self.config.model_type

            if model_type not in self.MODEL_TYPES:
                raise ValueError(f"Unsupported model type: {model_type}")

            try:
                # Dynamic import of appropriate model implementation
                logger.info(f"Loading {model_type} model: {self.config.model_name}")

                # Check if module exists
                module_path = self.MODEL_TYPES[model_type]
                spec = importlib.util.find_spec(module_path)

                if not spec:
                    # If module doesn't exist, use fallback implementation
                    logger.warning(
                        f"Model module {module_path} not found, using fallback implementation"
                    )
                    self.model = SimpleFallbackModel(self.config)
                else:
                    # Import module and create model
                    model_module = importlib.import_module(module_path)
                    model_class = getattr(
                        model_module, f"{model_type.capitalize()}Model"
                    )
                    self.model = model_class(self.config)

                self.model_loaded = True
                logger.info(f"Model {self.config.model_name} loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load model {self.config.model_name}: {e}")
                # Provide fallback model to avoid crashing
                self.model = SimpleFallbackModel(self.config)
                self.model_loaded = True

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate completion for a prompt.

        Args:
            prompt: Input text
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        if not self.model_loaded:
            self.load_model()

        return self.model.generate(
            prompt=prompt,
            max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
            temperature=(
                temperature if temperature is not None else self.config.temperature
            ),
        )

    def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        """
        Generate streaming completion for a prompt.

        Args:
            prompt: Input text
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature

        Returns:
            Iterator of generated text chunks
        """
        if not self.model_loaded:
            self.load_model()

        return self.model.generate_stream(
            prompt=prompt,
            max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
            temperature=(
                temperature if temperature is not None else self.config.temperature
            ),
        )

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text into token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        if not self.model_loaded:
            self.load_model()

        return self.model.tokenize(text)

    def num_tokens(self, text: str) -> int:
        """
        Count the number of tokens in text.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        if not self.model_loaded:
            self.load_model()

        return self.model.num_tokens(text)

    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self.model, "close") and callable(getattr(self.model, "close")):
            self.model.close()
        self.model = None
        self.model_loaded = False
        logger.info("Model unloaded and resources released")


class SimpleFallbackModel(ModelInterface):
    """
    A simple fallback model for when actual models can't be loaded.

    This provides simple, hardcoded responses to avoid crashing when
    model loading fails. Useful for testing or on systems without ML libs.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize fallback model.

        Args:
            config: Model configuration (ignored but kept for interface)
        """
        self.config = config
        logger.warning(
            "Using fallback model - actual language models could not be loaded"
        )

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate a simple fallback response.

        Args:
            prompt: Input text (used for simple pattern matching)
            max_tokens: Maximum number of tokens to generate (ignored)
            temperature: Sampling temperature (ignored)

        Returns:
            Generated text
        """
        prompt_lower = prompt.lower()

        if "hello" in prompt_lower or "hi" in prompt_lower:
            return "Hello! I'm a fallback model because the main language models couldn't be loaded."

        if "who are you" in prompt_lower or "your name" in prompt_lower:
            return "I am the Eidosian Forge, but currently running in fallback mode."

        if "?" in prompt:
            return "I'm not able to answer questions properly in fallback mode. Please check your model configuration."

        return (
            "I'm operating in fallback mode as the language model couldn't be loaded. "
            "Please check your configuration and dependencies."
        )

    def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        """
        Generate a simple fallback response in a stream.

        Args:
            prompt: Input text (used for simple pattern matching)
            max_tokens: Maximum number of tokens to generate (ignored)
            temperature: Sampling temperature (ignored)

        Returns:
            Iterator of generated text chunks
        """
        response = self.generate(prompt, max_tokens, temperature)

        # Simulate streaming by yielding a few chunks
        words = response.split()
        chunks = [" ".join(words[i : i + 3]) for i in range(0, len(words), 3)]

        for chunk in chunks:
            yield chunk + " "

    def tokenize(self, text: str) -> List[int]:
        """
        Fake tokenization by simply using character codes.

        Args:
            text: Input text

        Returns:
            List of "token" IDs (actually character codes)
        """
        # Super simplified "tokenization"
        return [ord(c) for c in text]

    def num_tokens(self, text: str) -> int:
        """
        Count the number of "tokens" (characters in fallback).

        Args:
            text: Input text

        Returns:
            Number of characters
        """
        return len(text)


# Factory function for creating model implementations
def create_model_manager(config: ModelConfig) -> ModelManager:
    """
    Create a model manager with the specified configuration.

    Args:
        config: Model configuration

    Returns:
        Configured ModelManager instance
    """
    return ModelManager(config)
