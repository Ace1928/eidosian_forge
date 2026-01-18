from dataclasses import dataclass
from typing import Any, List, Optional
from .base import BaseInferenceType
@dataclass
class TextToImageParameters(BaseInferenceType):
    """Additional inference parameters
    Additional inference parameters for Text To Image
    """
    guidance_scale: Optional[float] = None
    'For diffusion models. A higher guidance scale value encourages the model to generate\n    images closely linked to the text prompt at the expense of lower image quality.\n    '
    negative_prompt: Optional[List[str]] = None
    'One or several prompt to guide what NOT to include in image generation.'
    num_inference_steps: Optional[int] = None
    'For diffusion models. The number of denoising steps. More denoising steps usually lead to\n    a higher quality image at the expense of slower inference.\n    '
    scheduler: Optional[str] = None
    'For diffusion models. Override the scheduler with a compatible one'
    target_size: Optional[TextToImageTargetSize] = None
    'The size in pixel of the output image'