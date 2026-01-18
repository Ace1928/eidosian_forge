import numbers
from typing import TYPE_CHECKING, Optional, Type, Union
import wandb
from wandb import util
from wandb.util import has_num
from ..base_types.json_metadata import JSONMetadata
Initialize a BoundingBoxes object.

        The input dictionary `val` should contain the keys:
            box_data: a list of dictionaries, each of which describes a bounding box.
            class_labels: (optional) A map of integer class labels to their readable
                class names.

        Each bounding box dictionary should contain the following keys:
            position: (dictionary) the position and size of the bounding box.
            domain: (string) One of two options for the bounding box coordinate domain.
            class_id: (integer) The class label id for this box.
            scores: (dictionary of string to number, optional) A mapping of named fields
                to numerical values (float or int).
            box_caption: (optional) The label text, often composed of the class label,
                class name, and/or scores.

        The position dictionary should be in one of two formats:
            {"minX", "minY", "maxX", "maxY"}: (dictionary) A set of coordinates defining
                the upper and lower bounds of the box (the bottom left and top right
                corners).
            {"middle", "width", "height"}: (dictionary) A set of coordinates defining
                the center and dimensions of the box, with "middle" as a list [x, y] for
                the center point and "width" and "height" as numbers.
        Note that boxes need not all use the same format.

        Args:
            val: (dictionary) A dictionary containing the bounding box data.
            key: (string) The readable name or id for this set of bounding boxes (e.g.
                predictions, ground_truth)
        