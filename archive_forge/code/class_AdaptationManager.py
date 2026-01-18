from heapq import heappop, heappush
import inspect
import itertools
import functools
from traits.adaptation.adaptation_error import AdaptationError
from traits.has_traits import HasTraits
from traits.trait_types import Dict, List, Str
class AdaptationManager(HasTraits):
    """ Manages all registered adaptations. """

    @staticmethod
    def mro_distance_to_protocol(from_type, to_protocol):
        """ Return the distance in the MRO from 'from_type' to 'to_protocol'.

        If `from_type` provides `to_protocol`, returns the distance between
        `from_type` and the super-most class in the MRO hierarchy providing
        `to_protocol` (that's where the protocol was provided in the first
        place).

        If `from_type` does not provide `to_protocol`, return None.

        """
        if not AdaptationManager.provides_protocol(from_type, to_protocol):
            return None
        supertypes = inspect.getmro(from_type)[1:]
        distance = 0
        for t in supertypes:
            if AdaptationManager.provides_protocol(t, to_protocol):
                distance += 1
            else:
                break
        return distance

    @staticmethod
    def provides_protocol(type_, protocol):
        """ Does the given type provide (i.e implement) a given protocol?

        Parameters
        ----------
        type_
            Python 'type'.
        protocol
            Either a regular Python class or a traits Interface.

        Returns
        -------
        result : bool
            True if the object provides the protocol, otherwise False.

        """
        return issubclass(type_, protocol)

    def adapt(self, adaptee, to_protocol, default=AdaptationError):
        """ Attempt to adapt an object to a given protocol.

        If *adaptee* already provides (i.e. implements) the given protocol
        then it is simply returned unchanged.

        Otherwise, we try to build a chain of adapters that adapt *adaptee* to
        *to_protocol*. If no such adaptation is possible then either an
        AdaptationError is raised, or *default* is returned.

        Parameters
        ----------
        adaptee : object
            The object that we want to adapt.
        to_protocol : type or interface
            The protocol that the want to adapt *adaptee* to.
        default : object, optional
            Object to return if no adaptation is possible. If no default is
            provided, and adaptation fails, an ``AdaptationError`` is raised.

        Returns
        -------
        adapted_object : to_protocol
            The original adaptee adapted to the target protocol.

        Raises
        ------
        AdaptationError
            If adaptation is not possible, and no default is given.
        """
        if self.provides_protocol(type(adaptee), to_protocol):
            result = adaptee
        else:
            result = self._adapt(adaptee, to_protocol)
        if result is None:
            if default is AdaptationError:
                raise AdaptationError('Could not adapt %r to %r' % (adaptee, to_protocol))
            else:
                result = default
        return result

    def register_offer(self, offer):
        """ Register an offer to adapt from one protocol to another. """
        offers = self._adaptation_offers.setdefault(offer.from_protocol_name, [])
        offers.append(offer)

    def register_factory(self, factory, from_protocol, to_protocol):
        """ Register an adapter factory.

        This is a simply a convenience method that creates and registers an
        'AdaptationOffer' from the given arguments.

        """
        from traits.adaptation.adaptation_offer import AdaptationOffer
        self.register_offer(AdaptationOffer(factory=factory, from_protocol=from_protocol, to_protocol=to_protocol))

    def register_provides(self, provider_protocol, protocol):
        """ Register that a protocol provides another. """
        self.register_factory(no_adapter_necessary, provider_protocol, protocol)

    def supports_protocol(self, obj, protocol):
        """ Does the object support a given protocol?

        An object "supports" a protocol if either it "provides" it directly,
        or it can be adapted to it.

        """
        return self.adapt(obj, protocol, None) is not None
    _adaptation_offers = Dict(Str, List)

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