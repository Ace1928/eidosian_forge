from heapq import heappop, heappush
import inspect
import itertools
import functools
from traits.adaptation.adaptation_error import AdaptationError
from traits.has_traits import HasTraits
from traits.trait_types import Dict, List, Str
def _by_weight_then_from_protocol_specificity(edge_1, edge_2):
    """ Comparison function for graph edges.

    Each edge is of the form (mro distance, adaptation offer).

    Comparison is done by mro distance first, and by offer's from_protocol
    issubclass next.

    If two edges have the same mro distance, and the from_protocols of the
    two edges are not subclasses of one another, they are considered "equal".

    """
    mro_distance_1, offer_1 = edge_1
    mro_distance_2, offer_2 = edge_2
    if mro_distance_1 < mro_distance_2:
        return -1
    elif mro_distance_1 > mro_distance_2:
        return 1
    if offer_1.from_protocol is offer_2.from_protocol:
        return 0
    if issubclass(offer_1.from_protocol, offer_2.from_protocol):
        return -1
    elif issubclass(offer_2.from_protocol, offer_1.from_protocol):
        return 1
    return 0