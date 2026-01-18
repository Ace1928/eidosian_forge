from typing import Any, Callable, List
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Extra
from langchain_community.llms.self_hosted import SelfHostedPipeline
def _embed_documents(pipeline: Any, *args: Any, **kwargs: Any) -> List[List[float]]:
    """Inference function to send to the remote hardware.

    Accepts a sentence_transformer model_id and
    returns a list of embeddings for each document in the batch.
    """
    return pipeline(*args, **kwargs)