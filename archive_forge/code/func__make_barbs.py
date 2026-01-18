import math
import numpy as np
from numpy import ma
from matplotlib import _api, cbook, _docstring
import matplotlib.artist as martist
import matplotlib.collections as mcollections
from matplotlib.patches import CirclePolygon
import matplotlib.text as mtext
import matplotlib.transforms as transforms
def _make_barbs(self, u, v, nflags, nbarbs, half_barb, empty_flag, length, pivot, sizes, fill_empty, flip):
    """
        Create the wind barbs.

        Parameters
        ----------
        u, v
            Components of the vector in the x and y directions, respectively.

        nflags, nbarbs, half_barb, empty_flag
            Respectively, the number of flags, number of barbs, flag for
            half a barb, and flag for empty barb, ostensibly obtained from
            :meth:`_find_tails`.

        length
            The length of the barb staff in points.

        pivot : {"tip", "middle"} or number
            The point on the barb around which the entire barb should be
            rotated.  If a number, the start of the barb is shifted by that
            many points from the origin.

        sizes : dict
            Coefficients specifying the ratio of a given feature to the length
            of the barb. These features include:

            - *spacing*: space between features (flags, full/half barbs).
            - *height*: distance from shaft of top of a flag or full barb.
            - *width*: width of a flag, twice the width of a full barb.
            - *emptybarb*: radius of the circle used for low magnitudes.

        fill_empty : bool
            Whether the circle representing an empty barb should be filled or
            not (this changes the drawing of the polygon).

        flip : list of bool
            Whether the features should be flipped to the other side of the
            barb (useful for winds in the southern hemisphere).

        Returns
        -------
        list of arrays of vertices
            Polygon vertices for each of the wind barbs.  These polygons have
            been rotated to properly align with the vector direction.
        """
    spacing = length * sizes.get('spacing', 0.125)
    full_height = length * sizes.get('height', 0.4)
    full_width = length * sizes.get('width', 0.25)
    empty_rad = length * sizes.get('emptybarb', 0.15)
    pivot_points = dict(tip=0.0, middle=-length / 2.0)
    endx = 0.0
    try:
        endy = float(pivot)
    except ValueError:
        endy = pivot_points[pivot.lower()]
    angles = -(ma.arctan2(v, u) + np.pi / 2)
    circ = CirclePolygon((0, 0), radius=empty_rad).get_verts()
    if fill_empty:
        empty_barb = circ
    else:
        empty_barb = np.concatenate((circ, circ[::-1]))
    barb_list = []
    for index, angle in np.ndenumerate(angles):
        if empty_flag[index]:
            barb_list.append(empty_barb)
            continue
        poly_verts = [(endx, endy)]
        offset = length
        barb_height = -full_height if flip[index] else full_height
        for i in range(nflags[index]):
            if offset != length:
                offset += spacing / 2.0
            poly_verts.extend([[endx, endy + offset], [endx + barb_height, endy - full_width / 2 + offset], [endx, endy - full_width + offset]])
            offset -= full_width + spacing
        for i in range(nbarbs[index]):
            poly_verts.extend([(endx, endy + offset), (endx + barb_height, endy + offset + full_width / 2), (endx, endy + offset)])
            offset -= spacing
        if half_barb[index]:
            if offset == length:
                poly_verts.append((endx, endy + offset))
                offset -= 1.5 * spacing
            poly_verts.extend([(endx, endy + offset), (endx + barb_height / 2, endy + offset + full_width / 4), (endx, endy + offset)])
        poly_verts = transforms.Affine2D().rotate(-angle).transform(poly_verts)
        barb_list.append(poly_verts)
    return barb_list