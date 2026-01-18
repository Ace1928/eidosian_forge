from __future__ import annotations
import warnings
import numpy as np
from pymatgen.analysis.structure_prediction.substitution_probability import SubstitutionPredictor
from pymatgen.core import Element, Species
def _shannon_radii_from_cn(species_list, cn_roman, radius_to_compare=0):
    """
    Utility func to get Shannon radii for a particular coordination number.

    As the Shannon radii depends on charge state and coordination number,
    species without an entry for a particular coordination number will
    be skipped.

    Args:
        species_list (list): A list of Species to get the Shannon radii for.
        cn_roman (str): The coordination number as a roman numeral. See
            Species.get_shannon_radius for more details.
        radius_to_compare (float, optional): If set, the data will be returned
            with a "radii_diff" key, containing the difference between the
            shannon radii and this radius.

    Returns:
        list[dict]: The Shannon radii for all Species in species. Formatted
            as a list of dictionaries, with the keys:

            - "species": The species with charge state.
            - "radius": The Shannon radius for the species.
            - "radius_diff": The difference between the Shannon radius and the
                radius_to_compare optional argument.
    """
    shannon_radii = []
    for s in species_list:
        try:
            radius = s.get_shannon_radius(cn_roman)
            shannon_radii.append({'species': s, 'radius': radius, 'radii_diff': radius - radius_to_compare})
        except KeyError:
            pass
    return shannon_radii