import numpy as np
from .boundingregion import BoundingBox
from .util import datetime_types
class SheetCoordinateSystem:
    """
    Provides methods to allow conversion between sheet and matrix
    coordinates.
    """

    def __get_xdensity(self):
        return self.__xdensity

    def __get_ydensity(self):
        return self.__ydensity

    def __get_shape(self):
        return self.__shape
    xdensity = property(__get_xdensity, doc='\n        The spacing between elements in an underlying matrix\n        representation, in the x direction.')
    ydensity = property(__get_ydensity, doc='\n        The spacing between elements in an underlying matrix\n        representation, in the y direction.')
    shape = property(__get_shape)
    _time_unit = 'us'

    def __init__(self, bounds, xdensity, ydensity=None):
        """
        Store the bounds (as l,b,r,t in an array), xdensity, and
        ydensity.

        If ydensity is not specified, it is assumed that the specified
        xdensity is nominal and that the true xdensity should be
        calculated. The top and bottom bounds are adjusted so that the
        ydensity is equal to the xdensity.

        If both xdensity and ydensity are specified, these and the
        bounds are taken to be exact and are not adjusted.
        """
        if not ydensity:
            bounds, xdensity = self.__equalize_densities(bounds, xdensity)
        self.bounds = bounds
        self.__set_xdensity(xdensity)
        self.__set_ydensity(ydensity or xdensity)
        self.lbrt = np.array(bounds.lbrt())
        r1, r2, c1, c2 = Slice._boundsspec2slicespec(self.lbrt, self)
        self.__shape = (r2 - r1, c2 - c1)

    def __set_xdensity(self, density):
        self.__xdensity = density
        self.__xstep = 1.0 / density

    def __set_ydensity(self, density):
        self.__ydensity = density
        self.__ystep = 1.0 / density

    def __equalize_densities(self, nominal_bounds, nominal_density):
        """
        Calculate the true density along x, and adjust the top and
        bottom bounds so that the density along y will be equal.

        Returns (adjusted_bounds, true_density)
        """
        left, bottom, right, top = nominal_bounds.lbrt()
        width, height = (right - left, top - bottom)
        center_y = bottom + height / 2.0
        true_density = int(nominal_density * width) / float(width)
        n_cells = round(height * true_density, 0)
        adjusted_half_height = n_cells / true_density / 2.0
        return (BoundingBox(points=((left, center_y - adjusted_half_height), (right, center_y + adjusted_half_height))), true_density)

    def sheet2matrix(self, x, y):
        """
        Convert a point (x,y) in Sheet coordinates to continuous
        matrix coordinates.

        Returns (float_row,float_col), where float_row corresponds to
        y, and float_col to x.

        Valid for scalar or array x and y.

        Note about Bounds For a Sheet with
        BoundingBox(points=((-0.5,-0.5),(0.5,0.5))) and density=3,
        x=-0.5 corresponds to float_col=0.0 and x=0.5 corresponds to
        float_col=3.0.  float_col=3.0 is not inside the matrix
        representing this Sheet, which has the three columns
        (0,1,2). That is, x=-0.5 is inside the BoundingBox but x=0.5
        is outside. Similarly, y=0.5 is inside (at row 0) but y=-0.5
        is outside (at row 3) (it's the other way round for y because
        the matrix row index increases as y decreases).
        """
        xdensity = self.__xdensity
        if isinstance(x, np.ndarray) and x.dtype.kind == 'M' or isinstance(x, datetime_types):
            xdensity = np.timedelta64(int(round(1.0 / xdensity)), self._time_unit)
            float_col = (x - self.lbrt[0]) / xdensity
        else:
            float_col = (x - self.lbrt[0]) * xdensity
        ydensity = self.__ydensity
        if isinstance(y, np.ndarray) and y.dtype.kind == 'M' or isinstance(y, datetime_types):
            ydensity = np.timedelta64(int(round(1.0 / ydensity)), self._time_unit)
            float_row = (self.lbrt[3] - y) / ydensity
        else:
            float_row = (self.lbrt[3] - y) * ydensity
        return (float_row, float_col)

    def sheet2matrixidx(self, x, y):
        """
        Convert a point (x,y) in sheet coordinates to the integer row
        and column index of the matrix cell in which that point falls,
        given a bounds and density.  Returns (row,column).

        Note that if coordinates along the right or bottom boundary
        are passed into this function, the returned matrix coordinate
        of the boundary will be just outside the matrix, because the
        right and bottom boundaries are exclusive.

        Valid for scalar or array x and y.
        """
        r, c = self.sheet2matrix(x, y)
        r = np.floor(r)
        c = np.floor(c)
        if hasattr(r, 'astype'):
            return (r.astype(int), c.astype(int))
        else:
            return (int(r), int(c))

    def matrix2sheet(self, float_row, float_col):
        """
        Convert a floating-point location (float_row,float_col) in
        matrix coordinates to its corresponding location (x,y) in
        sheet coordinates.

        Valid for scalar or array float_row and float_col.

        Inverse of sheet2matrix().
        """
        xoffset = float_col * self.__xstep
        if isinstance(self.lbrt[0], datetime_types):
            xoffset = np.timedelta64(int(round(xoffset)), self._time_unit)
        x = self.lbrt[0] + xoffset
        yoffset = float_row * self.__ystep
        if isinstance(self.lbrt[3], datetime_types):
            yoffset = np.timedelta64(int(round(yoffset)), self._time_unit)
        y = self.lbrt[3] - yoffset
        return (x, y)

    def matrixidx2sheet(self, row, col):
        """
        Return (x,y) where x and y are the floating point coordinates
        of the *center* of the given matrix cell (row,col). If the
        matrix cell represents a 0.2 by 0.2 region, then the center
        location returned would be 0.1,0.1.

        NOTE: This is NOT the strict mathematical inverse of
        sheet2matrixidx(), because sheet2matrixidx() discards all but
        the integer portion of the continuous matrix coordinate.

        Valid only for scalar or array row and col.
        """
        x, y = self.matrix2sheet(row + 0.5, col + 0.5)
        if not isinstance(x, datetime_types):
            x = np.around(x, 10)
        if not isinstance(y, datetime_types):
            y = np.around(y, 10)
        return (x, y)

    def closest_cell_center(self, x, y):
        """
        Given arbitrary sheet coordinates, return the sheet coordinates
        of the center of the closest unit.
        """
        return self.matrixidx2sheet(*self.sheet2matrixidx(x, y))

    def sheetcoordinates_of_matrixidx(self):
        """
        Return x,y where x is a vector of sheet coordinates
        representing the x-center of each matrix cell, and y
        represents the corresponding y-center of the cell.
        """
        rows, cols = self.shape
        return self.matrixidx2sheet(np.arange(rows), np.arange(cols))