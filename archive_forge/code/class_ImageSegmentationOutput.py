from typing import TYPE_CHECKING, List, TypedDict
class ImageSegmentationOutput(TypedDict):
    """Dictionary containing information about a [`~InferenceClient.image_segmentation`] task. In practice, image segmentation returns a
    list of `ImageSegmentationOutput` with 1 item per mask.

    Args:
        label (`str`):
            The label corresponding to the mask.
        mask (`Image`):
            An Image object representing the mask predicted by the model.
        score (`float`):
            The score associated with the label for this mask.
    """
    label: str
    mask: 'Image'
    score: float