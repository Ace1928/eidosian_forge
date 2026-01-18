from __future__ import absolute_import, print_function
from ..palette import Palette
class Cubehelix(Palette):
    """
    Representation of a Cubehelix color map with matplotlib compatible
    views of the map.

    Parameters
    ----------
    name : str
    colors : list
        Colors as list of 0-255 RGB triplets.

    Attributes
    ----------
    name : str
    type : str
    number : int
        Number of colors in color map.
    colors : list
        Colors as list of 0-255 RGB triplets.
    hex_colors : list
    mpl_colors : list
    mpl_colormap : matplotlib LinearSegmentedColormap

    """
    url = url

    def __init__(self, name, colors):
        super(Cubehelix, self).__init__(name, palette_type, colors)

    @classmethod
    def make(cls, start=0.5, rotation=-1.5, gamma=1.0, start_hue=None, end_hue=None, sat=None, min_sat=1.2, max_sat=1.2, min_light=0.0, max_light=1.0, n=256.0, reverse=False, name='custom_cubehelix'):
        """
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
        """
        if not HAVE_NPY:
            raise RuntimeError('numpy not available.')
        if start_hue is not None and end_hue is not None:
            start = (start_hue / 360.0 - 1.0) * 3.0
            rotation = end_hue / 360.0 - start / 3.0 - 1.0
        lambd = np.linspace(min_light, max_light, n)
        lambd_gamma = lambd ** gamma
        phi = 2.0 * np.pi * (start / 3.0 + rotation * lambd)
        if sat is None:
            sat = np.linspace(min_sat, max_sat, n)
        amp = sat * lambd_gamma * (1.0 - lambd_gamma) / 2.0
        rot_matrix = np.array([[-0.14861, +1.78277], [-0.29227, -0.90649], [+1.97294, 0.0]])
        sin_cos = np.array([np.cos(phi), np.sin(phi)])
        rgb = (lambd_gamma + amp * np.dot(rot_matrix, sin_cos)).T * 255.0
        np.clip(rgb, 0.0, 255.0, out=rgb)
        if reverse:
            rgb = rgb[::-1, :]
        colors = rgb.astype(int).tolist()
        return cls(name, colors)