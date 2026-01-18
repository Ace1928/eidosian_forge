import inspect
from functools import wraps
from math import atan2
from math import pi as PI
from math import sqrt
from warnings import warn
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial.distance import pdist
from . import _moments
from ._find_contours import find_contours
from ._marching_cubes_lewiner import marching_cubes
from ._regionprops_utils import (
def _props_to_dict(regions, properties=('label', 'bbox'), separator='-'):
    """Convert image region properties list into a column dictionary.

    Parameters
    ----------
    regions : (K,) list
        List of RegionProperties objects as returned by :func:`regionprops`.
    properties : tuple or list of str, optional
        Properties that will be included in the resulting dictionary
        For a list of available properties, please see :func:`regionprops`.
        Users should remember to add "label" to keep track of region
        identities.
    separator : str, optional
        For non-scalar properties not listed in OBJECT_COLUMNS, each element
        will appear in its own column, with the index of that element separated
        from the property name by this separator. For example, the inertia
        tensor of a 2D region will appear in four columns:
        ``inertia_tensor-0-0``, ``inertia_tensor-0-1``, ``inertia_tensor-1-0``,
        and ``inertia_tensor-1-1`` (where the separator is ``-``).

        Object columns are those that cannot be split in this way because the
        number of columns would change depending on the object. For example,
        ``image`` and ``coords``.

    Returns
    -------
    out_dict : dict
        Dictionary mapping property names to an array of values of that
        property, one value per region. This dictionary can be used as input to
        pandas ``DataFrame`` to map property names to columns in the frame and
        regions to rows.

    Notes
    -----
    Each column contains either a scalar property, an object property, or an
    element in a multidimensional array.

    Properties with scalar values for each region, such as "eccentricity", will
    appear as a float or int array with that property name as key.

    Multidimensional properties *of fixed size* for a given image dimension,
    such as "centroid" (every centroid will have three elements in a 3D image,
    no matter the region size), will be split into that many columns, with the
    name {property_name}{separator}{element_num} (for 1D properties),
    {property_name}{separator}{elem_num0}{separator}{elem_num1} (for 2D
    properties), and so on.

    For multidimensional properties that don't have a fixed size, such as
    "image" (the image of a region varies in size depending on the region
    size), an object array will be used, with the corresponding property name
    as the key.

    Examples
    --------
    >>> from skimage import data, util, measure
    >>> image = data.coins()
    >>> label_image = measure.label(image > 110, connectivity=image.ndim)
    >>> proplist = regionprops(label_image, image)
    >>> props = _props_to_dict(proplist, properties=['label', 'inertia_tensor',
    ...                                              'inertia_tensor_eigvals'])
    >>> props  # doctest: +ELLIPSIS +SKIP
    {'label': array([ 1,  2, ...]), ...
     'inertia_tensor-0-0': array([  4.012...e+03,   8.51..., ...]), ...
     ...,
     'inertia_tensor_eigvals-1': array([  2.67...e+02,   2.83..., ...])}

    The resulting dictionary can be directly passed to pandas, if installed, to
    obtain a clean DataFrame:

    >>> import pandas as pd  # doctest: +SKIP
    >>> data = pd.DataFrame(props)  # doctest: +SKIP
    >>> data.head()  # doctest: +SKIP
       label  inertia_tensor-0-0  ...  inertia_tensor_eigvals-1
    0      1         4012.909888  ...                267.065503
    1      2            8.514739  ...                  2.834806
    2      3            0.666667  ...                  0.000000
    3      4            0.000000  ...                  0.000000
    4      5            0.222222  ...                  0.111111

    """
    out = {}
    n = len(regions)
    for prop in properties:
        r = regions[0]
        orig_prop = prop
        prop = PROPS.get(prop, prop)
        rp = getattr(r, prop)
        if prop in COL_DTYPES:
            dtype = COL_DTYPES[prop]
        else:
            func = r._extra_properties[prop]
            dtype = _infer_regionprop_dtype(func, intensity=r._intensity_image is not None, ndim=r.image.ndim)
        if np.isscalar(rp) or prop in OBJECT_COLUMNS or dtype is np.object_:
            column_buffer = np.empty(n, dtype=dtype)
            for i in range(n):
                column_buffer[i] = regions[i][prop]
            out[orig_prop] = np.copy(column_buffer)
        else:
            modified_props = []
            locs = []
            for ind in np.ndindex(np.shape(rp)):
                modified_props.append(separator.join(map(str, (orig_prop,) + ind)))
                locs.append(ind if len(ind) > 1 else ind[0])
            n_columns = len(locs)
            column_data = np.empty((n, n_columns), dtype=dtype)
            for k in range(n):
                rp = np.asarray(regions[k][prop])
                for i, loc in enumerate(locs):
                    column_data[k, i] = rp[loc]
            for i, modified_prop in enumerate(modified_props):
                out[modified_prop] = column_data[:, i]
    return out