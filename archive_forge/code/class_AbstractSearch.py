import itertools
from .. import debug, revision, trace
from ..graph import DictParentsProvider, Graph, invert_parent_map
from ..repository import AbstractSearchResult
class AbstractSearch:
    """A search that can be executed, producing a search result.

    :seealso: AbstractSearchResult
    """

    def execute(self):
        """Construct a network-ready search result from this search description.

        This may take some time to search repositories, etc.

        :return: A search result (an object that implements
            AbstractSearchResult's API).
        """
        raise NotImplementedError(self.execute)