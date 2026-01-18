from typing import TYPE_CHECKING, List, TypedDict
class ObjectDetectionOutput(TypedDict):
    """Dictionary containing information about a [`~InferenceClient.object_detection`] task.

    Args:
        label (`str`):
            The label corresponding to the detected object.
        box (`dict`):
            A dict response of bounding box coordinates of
            the detected object: xmin, ymin, xmax, ymax
        score (`float`):
            The score corresponding to the detected object.
    """
    label: str
    box: dict
    score: float