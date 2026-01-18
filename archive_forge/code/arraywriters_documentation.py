import numpy as np
from .casting import best_float, floor_exact, int_abs, shared_range, type_info
from .volumeutils import array_to_file, finite_range

        Now we know the slope, we need the intercept.  The intercept will be
        such that:

            (in_min - inter) / slope = out_min

        Solving for the intercept:

            inter = in_min - out_min * slope

        We can also flip the sign of the slope.  In that case we match the
        in_max to the out_min:

            (in_max - inter_flipped) / -slope = out_min
            inter_flipped = in_max + out_min * slope

        When we reconstruct the data, we're going to do:

            data = saved_data * slope + inter

        We can't change the range of the saved data (the whole range of the
        integer type) or the range of the output data (the values we input). We
        can change the intermediate values ``saved_data * slope`` by choosing
        the sign of the slope to match the in_min or in_max to the left or
        right end of the saved data range.

        If the out_dtype is signed int, then abs(out_min) = abs(out_max) + 1
        and the absolute value and therefore precision for values at the left
        and right of the saved data range are very similar (e.g. -128 * slope,
        127 * slope respectively).

        If the out_dtype is unsigned int, then the absolute value at the left
        is 0 and the precision is much higher than for the right end of the
        range (e.g. 0 * slope, 255 * slope).

        If the out_dtype is unsigned int then we choose the sign of the slope
        to match the smaller of the in_min, in_max to the zero end of the saved
        range.
        