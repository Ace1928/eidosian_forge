import importlib
import logging
from typing import Any, Callable, List, Optional
from langchain_community.embeddings.self_hosted import SelfHostedEmbeddings
class SelfHostedHuggingFaceInstructEmbeddings(SelfHostedHuggingFaceEmbeddings):
    """HuggingFace InstructEmbedding models on self-hosted remote hardware.

    Supported hardware includes auto-launched instances on AWS, GCP, Azure,
    and Lambda, as well as servers specified
    by IP address and SSH credentials (such as on-prem, or another
    cloud like Paperspace, Coreweave, etc.).

    To use, you should have the ``runhouse`` python package installed.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import SelfHostedHuggingFaceInstructEmbeddings
            import runhouse as rh
            model_name = "hkunlp/instructor-large"
            gpu = rh.cluster(name='rh-a10x', instance_type='A100:1')
            hf = SelfHostedHuggingFaceInstructEmbeddings(
                model_name=model_name, hardware=gpu)
    """
    model_id: str = DEFAULT_INSTRUCT_MODEL
    'Model name to use.'
    embed_instruction: str = DEFAULT_EMBED_INSTRUCTION
    'Instruction to use for embedding documents.'
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    'Instruction to use for embedding query.'
    model_reqs: List[str] = ['./', 'InstructorEmbedding', 'torch']
    'Requirements to install on hardware to inference the model.'

    def __init__(self, **kwargs: Any):
        """Initialize the remote inference function."""
        load_fn_kwargs = kwargs.pop('load_fn_kwargs', {})
        load_fn_kwargs['model_id'] = load_fn_kwargs.get('model_id', DEFAULT_INSTRUCT_MODEL)
        load_fn_kwargs['instruct'] = load_fn_kwargs.get('instruct', True)
        load_fn_kwargs['device'] = load_fn_kwargs.get('device', 0)
        super().__init__(load_fn_kwargs=load_fn_kwargs, **kwargs)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace instruct model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        instruction_pairs = []
        for text in texts:
            instruction_pairs.append([self.embed_instruction, text])
        embeddings = self.client(self.pipeline_ref, instruction_pairs)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace instruct model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        instruction_pair = [self.query_instruction, text]
        embedding = self.client(self.pipeline_ref, [instruction_pair])[0]
        return embedding.tolist()