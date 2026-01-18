 Adapt the call signature of TraitDict.notify to create an event.

    Parameters
    ----------
    trait_dict : traits.trait_dict_object.TraitDict
        The dict being mutated.
    removed : dict
        Items removed from the dict
    added : dict
        Items added to the dict
    changed : dict
        Old values for items updated on the dict.

    Returns
    -------
    DictChangeEvent
    