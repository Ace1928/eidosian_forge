import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import (
from mypy_extensions import trait
from black.comments import contains_pragma_comment
from black.lines import Line, append_leaves
from black.mode import Feature, Mode, Preview
from black.nodes import (
from black.rusty import Err, Ok, Result
from black.strings import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
class StringTransformer(ABC):
    """
    An implementation of the Transformer protocol that relies on its
    subclasses overriding the template methods `do_match(...)` and
    `do_transform(...)`.

    This Transformer works exclusively on strings (for example, by merging
    or splitting them).

    The following sections can be found among the docstrings of each concrete
    StringTransformer subclass.

    Requirements:
        Which requirements must be met of the given Line for this
        StringTransformer to be applied?

    Transformations:
        If the given Line meets all of the above requirements, which string
        transformations can you expect to be applied to it by this
        StringTransformer?

    Collaborations:
        What contractual agreements does this StringTransformer have with other
        StringTransfomers? Such collaborations should be eliminated/minimized
        as much as possible.
    """
    __name__: Final = 'StringTransformer'

    def __init__(self, line_length: int, normalize_strings: bool) -> None:
        self.line_length = line_length
        self.normalize_strings = normalize_strings

    @abstractmethod
    def do_match(self, line: Line) -> TMatchResult:
        """
        Returns:
            * Ok(string_indices) such that for each index, `line.leaves[index]`
              is our target string if a match was able to be made. For
              transformers that don't result in more lines (e.g. StringMerger,
              StringParenStripper), multiple matches and transforms are done at
              once to reduce the complexity.
              OR
            * Err(CannotTransform), if no match could be made.
        """

    @abstractmethod
    def do_transform(self, line: Line, string_indices: List[int]) -> Iterator[TResult[Line]]:
        """
        Yields:
            * Ok(new_line) where new_line is the new transformed line.
              OR
            * Err(CannotTransform) if the transformation failed for some reason. The
              `do_match(...)` template method should usually be used to reject
              the form of the given Line, but in some cases it is difficult to
              know whether or not a Line meets the StringTransformer's
              requirements until the transformation is already midway.

        Side Effects:
            This method should NOT mutate @line directly, but it MAY mutate the
            Line's underlying Node structure. (WARNING: If the underlying Node
            structure IS altered, then this method should NOT be allowed to
            yield an CannotTransform after that point.)
        """

    def __call__(self, line: Line, _features: Collection[Feature], _mode: Mode) -> Iterator[Line]:
        """
        StringTransformer instances have a call signature that mirrors that of
        the Transformer type.

        Raises:
            CannotTransform(...) if the concrete StringTransformer class is unable
            to transform @line.
        """
        if not any((leaf.type == token.STRING for leaf in line.leaves)):
            raise CannotTransform('There are no strings in this line.')
        match_result = self.do_match(line)
        if isinstance(match_result, Err):
            cant_transform = match_result.err()
            raise CannotTransform(f'The string transformer {self.__class__.__name__} does not recognize this line as one that it can transform.') from cant_transform
        string_indices = match_result.ok()
        for line_result in self.do_transform(line, string_indices):
            if isinstance(line_result, Err):
                cant_transform = line_result.err()
                raise CannotTransform('StringTransformer failed while attempting to transform string.') from cant_transform
            line = line_result.ok()
            yield line