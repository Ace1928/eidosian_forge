from . import xpktools
Predict the i->j NOE position based on self peak (diagonal) assignments.

    Parameters
    ----------
    peaklist : xprtools.Peaklist
        List of peaks from which to derive predictions
    originNuc : str
        Name of originating nucleus.
    originResNum : int
        Index of originating residue.
    detectedNuc : str
        Name of detected nucleus.

    toResNum : int
        Index of detected residue.

    Returns
    -------
    returnLine : str
        The .xpk file entry for the predicted crosspeak.

    Examples
    --------
    Using predictNOE(peaklist,"N15","H1",10,12)
    where peaklist is of the type xpktools.peaklist
    would generate a .xpk file entry for a crosspeak
    that originated on N15 of residue 10 and ended up
    as magnetization detected on the H1 nucleus of
    residue 12


    Notes
    =====
    The initial peaklist is assumed to be diagonal (self peaks only)
    and currently there is no checking done to insure that this
    assumption holds true.  Check your peaklist for errors and
    off diagonal peaks before attempting to use predictNOE.

    