from matplotlib.transforms import Bbox, TransformedBbox, Affine2D

    A function that needs to be called when figure dpi changes during the
    drawing (e.g., rasterizing).  It recovers the bbox and re-adjust it with
    the new dpi.
    