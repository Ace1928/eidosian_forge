from __future__ import absolute_import, print_function
from ..palette import Palette

        Create an arbitrary Cubehelix color palette from the algorithm.

        See http://adsabs.harvard.edu/abs/2011arXiv1108.5083G for a technical
        explanation of the algorithm.

        Parameters
        ----------
        start : scalar, optional
            Sets the starting position in the RGB color space. 0=blue, 1=red,
            2=green. Default is ``0.5`` (purple).
        rotation : scalar, optional
            The number of rotations through the rainbow. Can be positive
            or negative, indicating direction of rainbow. Negative values
            correspond to Blue->Red direction. Default is ``-1.5``.
        start_hue : scalar, optional
            Sets the starting color, ranging from [-360, 360]. Combined with
            `end_hue`, this parameter overrides ``start`` and ``rotation``.
            This parameter is based on the D3 implementation by @mbostock.
            Default is ``None``.
        end_hue : scalar, optional
            Sets the ending color, ranging from [-360, 360]. Combined with
            `start_hue`, this parameter overrides ``start`` and ``rotation``.
            This parameter is based on the D3 implementation by @mbostock.
            Default is ``None``.
        gamma : scalar, optional
            The gamma correction for intensity. Values of ``gamma < 1``
            emphasize low intensities while ``gamma > 1`` emphasises high
            intensities. Default is ``1.0``.
        sat : scalar, optional
            The uniform saturation intensity factor. ``sat=0`` produces
            grayscale, while ``sat=1`` retains the full saturation. Setting
            ``sat>1`` oversaturates the color map, at the risk of clipping
            the color scale. Note that ``sat`` overrides both ``min_stat``
            and ``max_sat`` if set.
        min_sat : scalar, optional
            Saturation at the minimum level. Default is ``1.2``.
        max_sat : scalar, optional
            Satuation at the maximum level. Default is ``1.2``.
        min_light : scalar, optional
            Minimum lightness value. Default is ``0``.
        max_light : scalar, optional
            Maximum lightness value. Default is ``1``.
        n : scalar, optional
            Number of discrete rendered colors. Default is ``256``.
        reverse : bool, optional
            Set to ``True`` to reverse the color map. Will go from black to
            white. Good for density plots where shade -> density.
            Default is ``False``.
        name : str, optional
            Name of the color map (defaults to ``'custom_cubehelix'``).

        Returns
        -------
        palette : `Cubehelix`
            A Cubehelix color palette.
        