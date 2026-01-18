import skimage.io as io
def FITSFactory(image_ext):
    """Load an image extension from a FITS file and return a NumPy array

    Parameters
    ----------
    image_ext : tuple
        FITS extension to load, in the format ``(filename, ext_num)``.
        The FITS ``(extname, extver)`` format is unsupported, since this
        function is not called directly by the user and
        ``imread_collection()`` does the work of figuring out which
        extensions need loading.

    """
    if not isinstance(image_ext, tuple):
        raise TypeError('Expected a tuple')
    if len(image_ext) != 2:
        raise ValueError('Expected a tuple of length 2')
    filename = image_ext[0]
    extnum = image_ext[1]
    if not (isinstance(filename, str) and isinstance(extnum, int)):
        raise ValueError('Expected a (filename, extension) tuple')
    with fits.open(filename) as hdulist:
        data = hdulist[extnum].data
    if data is None:
        raise RuntimeError(f'Extension {extnum} of {filename} has no data')
    return data