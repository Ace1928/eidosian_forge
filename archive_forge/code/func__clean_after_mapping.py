from __future__ import annotations
import numpy as np
from .arrayproxy import ArrayProxy
from .arraywriters import ArrayWriter, WriterError, get_slope_inter, make_array_writer
from .batteryrunners import Report
from .fileholders import copy_file_map
from .spatialimages import HeaderDataError, HeaderTypeError, SpatialHeader, SpatialImage
from .volumeutils import (
from .wrapstruct import LabeledWrapStruct
def _clean_after_mapping(self):
    """Set format-specific stuff after converting header from mapping

        This routine cleans up Analyze-type headers that have had their fields
        set from an Analyze map returned by the ``as_analyze_map`` method.
        Nifti 1 / 2, SPM Analyze, Analyze are all Analyze-type headers.
        Because this map can set fields that are illegal for particular
        subtypes of the Analyze header, this routine cleans these up before the
        resulting header is checked and returned.

        For example, a Nifti1 single (``.nii``) header has magic "n+1".
        Passing the nifti single header for conversion to a Nifti1Pair header
        using the ``as_analyze_map`` method will by default set the header
        magic to "n+1", when it should be "ni1" for the pair header.  This
        method is for that kind of case - so the specific header can set fields
        like magic correctly, even though the mapping has given a wrong value.
        """
    pass