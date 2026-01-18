from heapq import heappop, heappush
import inspect
import itertools
import functools
from traits.adaptation.adaptation_error import AdaptationError
from traits.has_traits import HasTraits
from traits.trait_types import Dict, List, Str
def _get_applicable_offers(self, current_protocol, path):
    """ Find all adaptation offers that can be applied to a protocol.

        Return all the applicable offers together with the number of steps
        up the MRO hierarchy that need to be taken from the protocol
        to the offer's from_protocol.
        The returned object is a list of tuples (mro_distance, offer) .

        In terms of our graph algorithm, we're looking for all outgoing edges
        from the current node.
        """
    edges = []
    for from_protocol_name, offers in self._adaptation_offers.items():
        from_protocol = offers[0].from_protocol
        mro_distance = self.mro_distance_to_protocol(current_protocol, from_protocol)
        if mro_distance is not None:
            for offer in offers:
                if offer not in path:
                    edges.append((mro_distance, offer))
    return edges