from typing import TYPE_CHECKING
from ..models.auto import AutoModelForVision2Seq
from ..utils import requires_backends
from .base import PipelineTool
class ImageCaptioningTool(PipelineTool):
    default_checkpoint = 'Salesforce/blip-image-captioning-base'
    description = 'This is a tool that generates a description of an image. It takes an input named `image` which should be the image to caption, and returns a text that contains the description in English.'
    name = 'image_captioner'
    model_class = AutoModelForVision2Seq
    inputs = ['image']
    outputs = ['text']

    def __init__(self, *args, **kwargs):
        requires_backends(self, ['vision'])
        super().__init__(*args, **kwargs)

    def encode(self, image: 'Image'):
        return self.pre_processor(images=image, return_tensors='pt')

    def forward(self, inputs):
        return self.model.generate(**inputs)

    def decode(self, outputs):
        return self.pre_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()