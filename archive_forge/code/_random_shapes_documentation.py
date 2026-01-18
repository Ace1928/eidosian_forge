import math
import numpy as np
from .draw import polygon as draw_polygon, disk as draw_disk, ellipse as draw_ellipse
from .._shared.utils import warn
Generate an image with random shapes, labeled with bounding boxes.

    The image is populated with random shapes with random sizes, random
    locations, and random colors, with or without overlap.

    Shapes have random (row, col) starting coordinates and random sizes bounded
    by `min_size` and `max_size`. It can occur that a randomly generated shape
    will not fit the image at all. In that case, the algorithm will try again
    with new starting coordinates a certain number of times. However, it also
    means that some shapes may be skipped altogether. In that case, this
    function will generate fewer shapes than requested.

    Parameters
    ----------
    image_shape : tuple
        The number of rows and columns of the image to generate.
    max_shapes : int
        The maximum number of shapes to (attempt to) fit into the shape.
    min_shapes : int, optional
        The minimum number of shapes to (attempt to) fit into the shape.
    min_size : int, optional
        The minimum dimension of each shape to fit into the image.
    max_size : int, optional
        The maximum dimension of each shape to fit into the image.
    num_channels : int, optional
        Number of channels in the generated image. If 1, generate monochrome
        images, else color images with multiple channels. Ignored if
        ``multichannel`` is set to False.
    shape : {rectangle, circle, triangle, ellipse, None} str, optional
        The name of the shape to generate or `None` to pick random ones.
    intensity_range : {tuple of tuples of uint8, tuple of uint8}, optional
        The range of values to sample pixel values from. For grayscale
        images the format is (min, max). For multichannel - ((min, max),)
        if the ranges are equal across the channels, and
        ((min_0, max_0), ... (min_N, max_N)) if they differ. As the
        function supports generation of uint8 arrays only, the maximum
        range is (0, 255). If None, set to (0, 254) for each channel
        reserving color of intensity = 255 for background.
    allow_overlap : bool, optional
        If `True`, allow shapes to overlap.
    num_trials : int, optional
        How often to attempt to fit a shape into the image before skipping it.
    rng : {`numpy.random.Generator`, int}, optional
        Pseudo-random number generator.
        By default, a PCG64 generator is used (see :func:`numpy.random.default_rng`).
        If `rng` is an int, it is used to seed the generator.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    image : uint8 array
        An image with the fitted shapes.
    labels : list
        A list of labels, one per shape in the image. Each label is a
        (category, ((r0, r1), (c0, c1))) tuple specifying the category and
        bounding box coordinates of the shape.

    Examples
    --------
    >>> import skimage.draw
    >>> image, labels = skimage.draw.random_shapes((32, 32), max_shapes=3)
    >>> image # doctest: +SKIP
    array([
       [[255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        ...,
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255]]], dtype=uint8)
    >>> labels # doctest: +SKIP
    [('circle', ((22, 18), (25, 21))),
     ('triangle', ((5, 6), (13, 13)))]
    