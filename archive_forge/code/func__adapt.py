from heapq import heappop, heappush
import inspect
import itertools
import functools
from traits.adaptation.adaptation_error import AdaptationError
from traits.has_traits import HasTraits
from traits.trait_types import Dict, List, Str
def _adapt(self, adaptee, to_protocol):
    """ Returns an adapter that adapts an object to the target class.

        Returns None if no such adapter exists.

        """
    counter = itertools.count()
    offer_queue = [((0, 0, next(counter)), [], type(adaptee))]
    while len(offer_queue) > 0:
        weight, path, current_protocol = heappop(offer_queue)
        edges = self._get_applicable_offers(current_protocol, path)
        edges.sort(key=functools.cmp_to_key(_by_weight_then_from_protocol_specificity))
        for mro_distance, offer in edges:
            new_path = path + [offer]
            if self.provides_protocol(offer.to_protocol, to_protocol):
                adapter = adaptee
                for offer in new_path:
                    adapter = offer.factory(adapter)
                    if adapter is None:
                        break
                else:
                    return adapter
            else:
                adapter_weight, mro_weight, _ = weight
                new_weight = (adapter_weight + 1, mro_weight + mro_distance, next(counter))
                heappush(offer_queue, (new_weight, new_path, offer.to_protocol))
    return None