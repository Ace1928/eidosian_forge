 Adapt the call signature of TraitSet.notify to create an event.

    Parameters
    ----------
    trait_set : traits.trait_set_object.TraitSet
        The set being mutated.
    removed : set
        Values removed from the set.
    added : set
        Values added to the set.

    Returns
    -------
    SetChangeEvent
    