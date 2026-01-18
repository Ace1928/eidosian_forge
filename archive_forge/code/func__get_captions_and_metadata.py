from io import BytesIO
from pathlib import Path
from typing import Any, List, Tuple, Union
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _get_captions_and_metadata(self, model: Any, processor: Any, image: Union[str, Path, bytes]) -> Tuple[str, dict]:
    """Helper function for getting the captions and metadata of an image."""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError('`PIL` package not found, please install with `pip install pillow`')
    image_source = image
    try:
        if isinstance(image, bytes):
            image = Image.open(BytesIO(image)).convert('RGB')
        elif isinstance(image, str) and (image.startswith('http://') or image.startswith('https://')):
            image = Image.open(requests.get(image, stream=True).raw).convert('RGB')
        else:
            image = Image.open(image).convert('RGB')
    except Exception:
        if isinstance(image_source, bytes):
            msg = 'Could not get image data from bytes'
        else:
            msg = f'Could not get image data for {image_source}'
        raise ValueError(msg)
    inputs = processor(image, 'an image of', return_tensors='pt')
    output = model.generate(**inputs)
    caption: str = processor.decode(output[0])
    if isinstance(image_source, bytes):
        metadata: dict = {'image_source': 'Image bytes provided'}
    else:
        metadata = {'image_path': str(image_source)}
    return (caption, metadata)