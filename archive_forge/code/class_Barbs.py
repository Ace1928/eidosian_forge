import math
import numpy as np
from numpy import ma
from matplotlib import _api, cbook, _docstring
import matplotlib.artist as martist
import matplotlib.collections as mcollections
from matplotlib.patches import CirclePolygon
import matplotlib.text as mtext
import matplotlib.transforms as transforms
class Barbs(mcollections.PolyCollection):
    """
    Specialized PolyCollection for barbs.

    The only API method is :meth:`set_UVC`, which can be used to
    change the size, orientation, and color of the arrows.  Locations
    are changed using the :meth:`set_offsets` collection method.
    Possibly this method will be useful in animations.

    There is one internal function :meth:`_find_tails` which finds
    exactly what should be put on the barb given the vector magnitude.
    From there :meth:`_make_barbs` is used to find the vertices of the
    polygon to represent the barb based on this information.
    """

    @_docstring.interpd
    def __init__(self, ax, *args, pivot='tip', length=7, barbcolor=None, flagcolor=None, sizes=None, fill_empty=False, barb_increments=None, rounding=True, flip_barb=False, **kwargs):
        """
        The constructor takes one required argument, an Axes
        instance, followed by the args and kwargs described
        by the following pyplot interface documentation:
        %(barbs_doc)s
        """
        self.sizes = sizes or dict()
        self.fill_empty = fill_empty
        self.barb_increments = barb_increments or dict()
        self.rounding = rounding
        self.flip = np.atleast_1d(flip_barb)
        transform = kwargs.pop('transform', ax.transData)
        self._pivot = pivot
        self._length = length
        if None in (barbcolor, flagcolor):
            kwargs['edgecolors'] = 'face'
            if flagcolor:
                kwargs['facecolors'] = flagcolor
            elif barbcolor:
                kwargs['facecolors'] = barbcolor
            else:
                kwargs.setdefault('facecolors', 'k')
        else:
            kwargs['edgecolors'] = barbcolor
            kwargs['facecolors'] = flagcolor
        if 'linewidth' not in kwargs and 'lw' not in kwargs:
            kwargs['linewidth'] = 1
        x, y, u, v, c = _parse_args(*args, caller_name='barbs')
        self.x = x
        self.y = y
        xy = np.column_stack((x, y))
        barb_size = self._length ** 2 / 4
        super().__init__([], (barb_size,), offsets=xy, offset_transform=transform, **kwargs)
        self.set_transform(transforms.IdentityTransform())
        self.set_UVC(u, v, c)

    def _find_tails(self, mag, rounding=True, half=5, full=10, flag=50):
        """
        Find how many of each of the tail pieces is necessary.

        Parameters
        ----------
        mag : `~numpy.ndarray`
            Vector magnitudes; must be non-negative (and an actual ndarray).
        rounding : bool, default: True
            Whether to round or to truncate to the nearest half-barb.
        half, full, flag : float, defaults: 5, 10, 50
            Increments for a half-barb, a barb, and a flag.

        Returns
        -------
        n_flags, n_barbs : int array
            For each entry in *mag*, the number of flags and barbs.
        half_flag : bool array
            For each entry in *mag*, whether a half-barb is needed.
        empty_flag : bool array
            For each entry in *mag*, whether nothing is drawn.
        """
        if rounding:
            mag = half * np.around(mag / half)
        n_flags, mag = divmod(mag, flag)
        n_barb, mag = divmod(mag, full)
        half_flag = mag >= half
        empty_flag = ~(half_flag | (n_flags > 0) | (n_barb > 0))
        return (n_flags.astype(int), n_barb.astype(int), half_flag, empty_flag)

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

    def set_UVC(self, U, V, C=None):
        self.u = ma.masked_invalid(U, copy=True).ravel()
        self.v = ma.masked_invalid(V, copy=True).ravel()
        if len(self.flip) == 1:
            flip = np.broadcast_to(self.flip, self.u.shape)
        else:
            flip = self.flip
        if C is not None:
            c = ma.masked_invalid(C, copy=True).ravel()
            x, y, u, v, c, flip = cbook.delete_masked_points(self.x.ravel(), self.y.ravel(), self.u, self.v, c, flip.ravel())
            _check_consistent_shapes(x, y, u, v, c, flip)
        else:
            x, y, u, v, flip = cbook.delete_masked_points(self.x.ravel(), self.y.ravel(), self.u, self.v, flip.ravel())
            _check_consistent_shapes(x, y, u, v, flip)
        magnitude = np.hypot(u, v)
        flags, barbs, halves, empty = self._find_tails(magnitude, self.rounding, **self.barb_increments)
        plot_barbs = self._make_barbs(u, v, flags, barbs, halves, empty, self._length, self._pivot, self.sizes, self.fill_empty, flip)
        self.set_verts(plot_barbs)
        if C is not None:
            self.set_array(c)
        xy = np.column_stack((x, y))
        self._offsets = xy
        self.stale = True

    def set_offsets(self, xy):
        """
        Set the offsets for the barb polygons.  This saves the offsets passed
        in and masks them as appropriate for the existing U/V data.

        Parameters
        ----------
        xy : sequence of pairs of floats
        """
        self.x = xy[:, 0]
        self.y = xy[:, 1]
        x, y, u, v = cbook.delete_masked_points(self.x.ravel(), self.y.ravel(), self.u, self.v)
        _check_consistent_shapes(x, y, u, v)
        xy = np.column_stack((x, y))
        super().set_offsets(xy)
        self.stale = True
    barbs_doc = _api.deprecated('3.7')(property(lambda self: _barbs_doc))