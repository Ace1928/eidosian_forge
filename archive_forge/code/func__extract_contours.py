import plotly.colors as clrs
from plotly.graph_objs import graph_objs as go
from plotly import exceptions
from plotly import optional_imports
from skimage import measure
def _extract_contours(im, values, colors):
    """
    Utility function for _contour_trace.

    In ``im`` only one part of the domain has valid values (corresponding
    to a subdomain where barycentric coordinates are well defined). When
    computing contours, we need to assign values outside of this domain.
    We can choose a value either smaller than all the values inside the
    valid domain, or larger. This value must be chose with caution so that
    no spurious contours are added. For example, if the boundary of the valid
    domain has large values and the outer value is set to a small one, all
    intermediate contours will be added at the boundary.

    Therefore, we compute the two sets of contours (with an outer value
    smaller of larger than all values in the valid domain), and choose
    the value resulting in a smaller total number of contours. There might
    be a faster way to do this, but it works...
    """
    mask_nan = np.isnan(im)
    im_min, im_max = (im[np.logical_not(mask_nan)].min(), im[np.logical_not(mask_nan)].max())
    zz_min = np.copy(im)
    zz_min[mask_nan] = 2 * im_min
    zz_max = np.copy(im)
    zz_max[mask_nan] = 2 * im_max
    all_contours1, all_values1, all_areas1, all_colors1 = ([], [], [], [])
    all_contours2, all_values2, all_areas2, all_colors2 = ([], [], [], [])
    for i, val in enumerate(values):
        contour_level1 = measure.find_contours(zz_min, val)
        contour_level2 = measure.find_contours(zz_max, val)
        all_contours1.extend(contour_level1)
        all_contours2.extend(contour_level2)
        all_values1.extend([val] * len(contour_level1))
        all_values2.extend([val] * len(contour_level2))
        all_areas1.extend([_polygon_area(contour.T[1], contour.T[0]) for contour in contour_level1])
        all_areas2.extend([_polygon_area(contour.T[1], contour.T[0]) for contour in contour_level2])
        all_colors1.extend([colors[i]] * len(contour_level1))
        all_colors2.extend([colors[i]] * len(contour_level2))
    if len(all_contours1) <= len(all_contours2):
        return (all_contours1, all_values1, all_areas1, all_colors1)
    else:
        return (all_contours2, all_values2, all_areas2, all_colors2)