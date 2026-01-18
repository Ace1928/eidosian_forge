import abc
import collections
import copy
import dataclasses
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from .arrow_reader import FileInstructions, make_file_instructions
from .naming import _split_re
from .utils.py_utils import NonMutableDict, asdict
@dataclass
class SplitGenerator:
    """Defines the split information for the generator.

    This should be used as returned value of
    `GeneratorBasedBuilder._split_generators`.
    See `GeneratorBasedBuilder._split_generators` for more info and example
    of usage.

    Args:
        name (`str`):
            Name of the `Split` for which the generator will
            create the examples.
        **gen_kwargs (additional keyword arguments):
            Keyword arguments to forward to the `DatasetBuilder._generate_examples` method
            of the builder.

    Example:

    ```py
    >>> datasets.SplitGenerator(
    ...     name=datasets.Split.TRAIN,
    ...     gen_kwargs={"split_key": "train", "files": dl_manager.download_and_extract(url)},
    ... )
    ```
    """
    name: str
    gen_kwargs: Dict = dataclasses.field(default_factory=dict)
    split_info: SplitInfo = dataclasses.field(init=False)

    def __post_init__(self):
        self.name = str(self.name)
        NamedSplit(self.name)
        self.split_info = SplitInfo(name=self.name)