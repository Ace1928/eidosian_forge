import os.path as op
import zipfile
import sys
from .schema import class_name
from . import uriutil

    Download all the files at this level that match the given constraint as a
    zip archive. Should not be called directly but from a instance of class
    that supports bulk downloading eg. "Scans"

        Parameters
        ----------
        instance : 'object
             The instance that contains local values needed by this function
             eg. instance._cbase stores the URI.
        dest_dir : string
             directory into which to place the downloaded archive.
        type: string
             a comma separated list of file types, eg. "T1,T2". Default is
             "ALL".
        name: string
             the name of the zip archive. Defaults to None. See below for the
             default naming scheme.
        extract: bool
             If True, the files are left extracted in the parent directory.
             Default is False.
        safe: bool
             If true, run safety checks on the files before extracting,
             eg. check that the file doesn't exist in the parent directory
             before overwriting it. Default is False.

        Default Zip Name
        ----------------
        Given the project "p", subject "s" and experiment "e", and that the
        "Scans" (as opposed to "Assessors" or "Reconstructions") are being
        downloaded, and the scan types are constrained to "T1,T2", the name of
        the zip file defaults to:
          "p_s_e_scans_T1_T2.zip"

        Exceptions
        ----------
        A generic Exception will be raised if any of the following happen:
         - This function is called directly and not from an instance of a
            class that supports bulk downloading eg."Scans"
         - The destination directory is unspecified
        A LookupError is raised if there are no resources to download
        A ValueError is raised if any of the following happen:
         - The project, subject and experiment names could not be extracted
            from the URI
         - The type constraint "ALL" is used with other constraints.
            eg. "ALL,T1,T2"
         - The URI associated with this class contains wildcards
            eg. /projects/proj/subjects/*/experiments/scans
        An EnvironmentError is raised if any of the following happen:
         - If "safe" is true, and (a) a zip file with the same name exists in
            given destination directory or
           (b) extracting the archive overrides an existing file. In the second
            case the downloaded archive
           is left in the parent directory.

        Return
        ------
        A path to the zip archive if "extract" is False, and a list of
        extracted files if True.
    