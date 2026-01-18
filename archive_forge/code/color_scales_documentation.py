from collections import OrderedDict
import random
Produces lookup table keyed by each class of data, with value as an RGB array

    Parameters
    ---------
    data_vector : list
        Vector of data classes to be categorized, passed from the data itself

    Returns
    -------
    collections.OrderedDict
        Dictionary of random RGBA value per class, keyed on class
    