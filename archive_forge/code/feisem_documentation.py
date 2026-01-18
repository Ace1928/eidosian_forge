from .tifffile import TiffFormat
Read the metadata from an FEI SEM TIFF.

            This metadata is included as ASCII text at the end of the file.

            The index, if provided, is ignored.

            Returns
            -------
            metadata : dict
                Dictionary of metadata.
            