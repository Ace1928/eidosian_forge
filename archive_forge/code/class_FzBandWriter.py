from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
class FzBandWriter(object):
    """
    Wrapper class for struct `fz_band_writer`. Not copyable or assignable.
    fz_band_writer
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    MONO = _mupdf.FzBandWriter_MONO
    COLOR = _mupdf.FzBandWriter_COLOR
    PNG = _mupdf.FzBandWriter_PNG
    PNM = _mupdf.FzBandWriter_PNM
    PAM = _mupdf.FzBandWriter_PAM
    PBM = _mupdf.FzBandWriter_PBM
    PKM = _mupdf.FzBandWriter_PKM
    PS = _mupdf.FzBandWriter_PS
    PSD = _mupdf.FzBandWriter_PSD

    def fz_close_band_writer(self):
        """
        Class-aware wrapper for `::fz_close_band_writer()`.
        	Finishes up the output and closes the band writer. After this
        	call no more headers or bands may be written.
        """
        return _mupdf.FzBandWriter_fz_close_band_writer(self)

    def fz_pdfocr_band_writer_set_progress(self, progress_fn, progress_arg):
        """
        Class-aware wrapper for `::fz_pdfocr_band_writer_set_progress()`.
        	Set the progress callback for a pdfocr bandwriter.
        """
        return _mupdf.FzBandWriter_fz_pdfocr_band_writer_set_progress(self, progress_fn, progress_arg)

    def fz_write_band(self, stride, band_height, samples):
        """
        Class-aware wrapper for `::fz_write_band()`.
        	Cause a band writer to write the next band
        	of data for an image.

        	stride: The byte offset from the first byte of the data
        	for a pixel to the first byte of the data for the same pixel
        	on the row below.

        	band_height: The number of lines in this band.

        	samples: Pointer to first byte of the data.
        """
        return _mupdf.FzBandWriter_fz_write_band(self, stride, band_height, samples)

    def fz_write_header(self, w, h, n, alpha, xres, yres, pagenum, cs, seps):
        """
        Class-aware wrapper for `::fz_write_header()`.
        	Cause a band writer to write the header for
        	a banded image with the given properties/dimensions etc. This
        	also configures the bandwriter for the format of the data to be
        	passed in future calls.

        	w, h: Width and Height of the entire page.

        	n: Number of components (including spots and alphas).

        	alpha: Number of alpha components.

        	xres, yres: X and Y resolutions in dpi.

        	cs: Colorspace (NULL for bitmaps)

        	seps: Separation details (or NULL).
        """
        return _mupdf.FzBandWriter_fz_write_header(self, w, h, n, alpha, xres, yres, pagenum, cs, seps)

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_new_band_writer_of_size()`.

        |

        *Overload 2:*
         Constructor using `fz_new_color_pcl_band_writer()`.
        		Create a new band writer, outputing color pcl.
        	 Constructor using `fz_new_mono_pcl_band_writer()`.
        		Create a new band writer, outputing monochrome pcl.


        |

        *Overload 3:*
         Constructor using `fz_new_mono_pwg_band_writer()`.
        		Create a new monochrome pwg band writer.


        |

        *Overload 4:*
         Constructor using `fz_new_pam_band_writer()`.
        		Create a band writer targetting pnm (greyscale, rgb or cmyk,
        		with or without alpha).
        	 Constructor using `fz_new_pbm_band_writer()`.
        		Create a new band writer, targetting pbm.
        	 Constructor using `fz_new_pclm_band_writer()`.
        		Create a new band writer, outputing pclm


        |

        *Overload 5:*
         Constructor using `fz_new_pdfocr_band_writer()`.
        		Create a new band writer, outputing pdfocr.

        		Ownership of output stays with the caller, the band writer
        		borrows the reference. The caller must keep the output around
        		for the duration of the band writer, and then close/drop as
        		appropriate.


        |

        *Overload 6:*
         Constructor using `fz_new_pkm_band_writer()`.
        		Create a new pkm band writer for CMYK pixmaps.
        	 Constructor using `fz_new_png_band_writer()`.
        		Create a new png band writer (greyscale or RGB, with or without
        		alpha).


        |

        *Overload 7:*
         Constructor using `fz_new_pnm_band_writer()`.
        		Create a band writer targetting pnm (greyscale or rgb, no
        		alpha).
        	 Constructor using `fz_new_ps_band_writer()`.
        		Create a postscript band writer for gray, rgb, or cmyk, no
        		alpha.
        	 Constructor using `fz_new_psd_band_writer()`.
        		Open a PSD band writer.
        	 Constructor using `fz_new_pwg_band_writer()`.
        		Create a new color pwg band writer.
        	 Constructor using fz_new_mono_pcl_band_writer() or fz_new_color_pcl_band_writer().

        |

        *Overload 8:*
         Constructor using fz_new_p*_band_writer().

        |

        *Overload 9:*
         Constructor using fz_new_mono_pwg_band_writer() or fz_new_pwg_band_writer().

        |

        *Overload 10:*
         Default constructor, sets `m_internal` to null.

        |

        *Overload 11:*
         Constructor using raw copy of pre-existing `::fz_band_writer`.
        """
        _mupdf.FzBandWriter_swiginit(self, _mupdf.new_FzBandWriter(*args))
    __swig_destroy__ = _mupdf.delete_FzBandWriter

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzBandWriter_m_internal_value(self)
    m_internal = property(_mupdf.FzBandWriter_m_internal_get, _mupdf.FzBandWriter_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzBandWriter_s_num_instances_get, _mupdf.FzBandWriter_s_num_instances_set)