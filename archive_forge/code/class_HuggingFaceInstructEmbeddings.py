from typing import Any, Dict, List, Optional
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, SecretStr
class HuggingFaceInstructEmbeddings(BaseModel, Embeddings):
    """Wrapper around sentence_transformers embedding models.

    To use, you should have the ``sentence_transformers``
    and ``InstructorEmbedding`` python packages installed.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import HuggingFaceInstructEmbeddings

            model_name = "hkunlp/instructor-large"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': True}
            hf = HuggingFaceInstructEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
    """
    client: Any
    model_name: str = DEFAULT_INSTRUCT_MODEL
    'Model name to use.'
    cache_folder: Optional[str] = None
    'Path to store models. \n    Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable.'
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    'Keyword arguments to pass to the model.'
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    'Keyword arguments to pass when calling the `encode` method of the model.'
    embed_instruction: str = DEFAULT_EMBED_INSTRUCTION
    'Instruction to use for embedding documents.'
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    'Instruction to use for embedding query.'

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)
        try:
            from InstructorEmbedding import INSTRUCTOR
            self.client = INSTRUCTOR(self.model_name, cache_folder=self.cache_folder, **self.model_kwargs)
        except ImportError as e:
            raise ImportError('Dependencies for InstructorEmbedding not found.') from e

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace instruct model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        instruction_pairs = [[self.embed_instruction, text] for text in texts]
        embeddings = self.client.encode(instruction_pairs, **self.encode_kwargs)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace instruct model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        instruction_pair = [self.query_instruction, text]
        embedding = self.client.encode([instruction_pair], **self.encode_kwargs)[0]
        return embedding.tolist()